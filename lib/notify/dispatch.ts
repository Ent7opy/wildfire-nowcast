/**
 * Stage 4 notification dispatcher.
 *
 * Single entry point: `dispatchBrief(db, briefId, deps?)`. Called once per
 * brief that Stage 3 just persisted.
 *
 * Steps (mirrors `pm/briefs/18-stage4-notification-dispatch.md` §Dispatcher):
 *   1. Load brief + AOI + rules + user email.
 *   2. Resolve channels (rules.notify_channels, fallback to users.email).
 *   3. Per-channel: idempotency check → pause/quiet-hours gate → send → persist.
 *
 * Webhook channels are persisted as `skipped, channel_not_implemented`
 * (Stage 6 owns Slack/Discord delivery). RESEND_API_KEY missing →
 * persisted as `config_missing`; the route warns once per poll.
 *
 * Two-backend repository pattern: every SQL touched is non-spatial and runs
 * identically on Neon+PostGIS and PGlite.
 */
import { createHash } from "node:crypto";
import { sql } from "drizzle-orm";
import type { AppDb } from "@/lib/db/client";
import { sendEmail, type SendResult } from "./resend";

export type DispatchAttempt =
  | { status: "sent"; channel: string; target: string; providerMessageId: string }
  | { status: "failed"; channel: string; target: string; error: string }
  | { status: "skipped"; channel: string; target: string; reason: string }
  | { status: "config_missing"; channel: string; target: string };

export type DispatchOutcome = {
  briefId: string;
  attempts: DispatchAttempt[];
};

export type DispatchDeps = {
  send?: (args: {
    to: string;
    subject: string;
    markdown: string;
  }) => Promise<SendResult>;
  now?: Date;
};

type LoadedBrief = {
  briefId: string;
  aoiId: string;
  aoiName: string;
  renderedMarkdown: string;
  summary: string;
  pausedUntil: Date | null;
  quietHours: { tz: string; startHour: number; endHour: number } | null;
  notifyChannels: Array<
    | { type: "email"; target: string }
    | { type: "webhook"; target: string }
  >;
  userEmail: string | null;
};

export async function dispatchBrief(
  db: AppDb,
  briefId: string,
  deps: DispatchDeps = {},
): Promise<DispatchOutcome> {
  const loaded = await loadBrief(db, briefId);
  if (!loaded) {
    return { briefId, attempts: [] };
  }

  const now = deps.now ?? new Date();
  const send = deps.send ?? ((a) => sendEmail(a));

  const channels = resolveChannels(loaded);
  const attempts: DispatchAttempt[] = [];
  let firstSuccessRecorded = false;

  if (channels.length === 0) {
    const reason = isPendingPlaceholder(loaded.userEmail)
      ? "no_recipient_pending"
      : "no_recipient";
    const target = loaded.userEmail ?? "";
    await persistAttempt(db, {
      aoiId: loaded.aoiId,
      briefId,
      channel: "email",
      target,
      status: "skipped",
      skipReason: reason,
    });
    attempts.push({
      status: "skipped",
      channel: "email",
      target,
      reason,
    });
    return { briefId, attempts };
  }

  for (const ch of channels) {
    if (ch.type === "webhook") {
      const webhookHash = sha256(ch.target);
      const existingWebhook = await findExistingTerminalRow(
        db,
        briefId,
        "webhook",
        webhookHash,
      );
      if (existingWebhook) {
        attempts.push({
          status: "skipped",
          channel: "webhook",
          target: ch.target,
          reason: "duplicate",
        });
        continue;
      }
      await persistAttempt(db, {
        aoiId: loaded.aoiId,
        briefId,
        channel: "webhook",
        target: ch.target,
        targetHash: webhookHash,
        status: "skipped",
        skipReason: "channel_not_implemented",
      });
      attempts.push({
        status: "skipped",
        channel: "webhook",
        target: ch.target,
        reason: "channel_not_implemented",
      });
      continue;
    }

    const target = ch.target;
    const targetHash = sha256(target);

    const existing = await findExistingTerminalRow(db, briefId, "email", targetHash);
    if (existing) {
      attempts.push({
        status: "skipped",
        channel: "email",
        target,
        reason: "duplicate",
      });
      continue;
    }

    if (loaded.pausedUntil && loaded.pausedUntil.getTime() > now.getTime()) {
      await persistAttempt(db, {
        aoiId: loaded.aoiId,
        briefId,
        channel: "email",
        target,
        targetHash,
        status: "skipped",
        skipReason: "paused",
      });
      attempts.push({
        status: "skipped",
        channel: "email",
        target,
        reason: "paused",
      });
      continue;
    }

    // TODO Stage 6: hold + release as a morning digest at the top of the
    // quiet window per US-2 acceptance #3. Stage 4 is skip-only.
    if (loaded.quietHours && inQuietHours(now, loaded.quietHours)) {
      await persistAttempt(db, {
        aoiId: loaded.aoiId,
        briefId,
        channel: "email",
        target,
        targetHash,
        status: "skipped",
        skipReason: "quiet_hours",
      });
      attempts.push({
        status: "skipped",
        channel: "email",
        target,
        reason: "quiet_hours",
      });
      continue;
    }

    const subject = buildSubject(loaded.summary);
    const result = await send({
      to: target,
      subject,
      markdown: loaded.renderedMarkdown,
    });

    if (!result.ok && result.code === "config_missing") {
      await persistAttempt(db, {
        aoiId: loaded.aoiId,
        briefId,
        channel: "email",
        target,
        targetHash,
        status: "config_missing",
      });
      attempts.push({ status: "config_missing", channel: "email", target });
      continue;
    }

    if (!result.ok) {
      await persistAttempt(db, {
        aoiId: loaded.aoiId,
        briefId,
        channel: "email",
        target,
        targetHash,
        status: "failed",
        error: truncate(`${result.code}: ${result.message}`, 500),
      });
      attempts.push({
        status: "failed",
        channel: "email",
        target,
        error: `${result.code}: ${result.message}`,
      });
      continue;
    }

    const updateLastNotified = !firstSuccessRecorded;
    await persistSuccess(db, {
      aoiId: loaded.aoiId,
      briefId,
      target,
      targetHash,
      providerMessageId: result.providerMessageId,
      updateLastNotified,
    });
    firstSuccessRecorded = true;
    attempts.push({
      status: "sent",
      channel: "email",
      target,
      providerMessageId: result.providerMessageId,
    });
  }

  return { briefId, attempts };
}

function resolveChannels(loaded: LoadedBrief): Array<
  | { type: "email"; target: string }
  | { type: "webhook"; target: string }
> {
  if (loaded.notifyChannels.length > 0) {
    return loaded.notifyChannels.filter(
      (c) => c.type !== "email" || !isPendingPlaceholder(c.target),
    );
  }
  if (loaded.userEmail && !isPendingPlaceholder(loaded.userEmail)) {
    return [{ type: "email", target: loaded.userEmail }];
  }
  return [];
}

function isPendingPlaceholder(email: string | null | undefined): boolean {
  return typeof email === "string" && /@pending\.invalid$/.test(email);
}

function buildSubject(summary: string): string {
  return truncate(summary, 90);
}

function truncate(s: string, max: number): string {
  return s.length <= max ? s : s.slice(0, max);
}

function sha256(s: string): string {
  return createHash("sha256").update(s).digest("hex");
}

/**
 * Quiet-hours check — straightforward [startHour, endHour) window in the
 * configured tz. Wraparound (e.g. start=22, end=7) is supported via OR.
 *
 * Uses `Intl.DateTimeFormat` to read the hour in the AOI's tz; this is the
 * same tz handling the Stage 6 digest will need, kept tiny for Stage 4.
 */
export function inQuietHours(
  now: Date,
  qh: { tz: string; startHour: number; endHour: number },
): boolean {
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: qh.tz,
    hour: "numeric",
    hour12: false,
  });
  const parts = fmt.formatToParts(now);
  const hourPart = parts.find((p) => p.type === "hour");
  if (!hourPart) return false;
  const hour = Number(hourPart.value) % 24;
  if (qh.startHour === qh.endHour) return false;
  if (qh.startHour < qh.endHour) {
    return hour >= qh.startHour && hour < qh.endHour;
  }
  return hour >= qh.startHour || hour < qh.endHour;
}

// ---------------------------------------------------------------------------
// DB load / persist

async function loadBrief(db: AppDb, briefId: string): Promise<LoadedBrief | null> {
  const result = (await db.execute(sql`
    SELECT
      b."id"                AS brief_id,
      b."aoi_id"            AS aoi_id,
      b."rendered_markdown" AS rendered_markdown,
      b."payload"           AS payload,
      a."name"              AS aoi_name,
      a."user_id"           AS user_id,
      r."paused_until"      AS paused_until,
      r."quiet_hours"       AS quiet_hours,
      r."notify_channels"   AS notify_channels,
      u."email"             AS user_email
    FROM "aoi_briefs" b
    JOIN "aois" a       ON a."id" = b."aoi_id"
    LEFT JOIN "aoi_rules" r ON r."aoi_id" = b."aoi_id"
    LEFT JOIN "users" u     ON u."id" = a."user_id"
    WHERE b."id" = ${briefId}
    LIMIT 1
  `)) as unknown as { rows?: Array<Record<string, unknown>> };
  const rows = (result.rows ??
    (result as unknown as Array<Record<string, unknown>>)) as Array<
    Record<string, unknown>
  >;
  const row = rows[0];
  if (!row) return null;

  const payload = row.payload as { summary?: string } | string | null;
  let summary = "";
  if (typeof payload === "string") {
    try {
      const parsed = JSON.parse(payload) as { summary?: string };
      summary = parsed?.summary ?? "";
    } catch {
      summary = "";
    }
  } else if (payload && typeof payload === "object") {
    summary = typeof payload.summary === "string" ? payload.summary : "";
  }

  const channelsRaw = row.notify_channels;
  const channels = parseChannels(channelsRaw);
  const quietHours = parseQuietHours(row.quiet_hours);

  return {
    briefId: String(row.brief_id),
    aoiId: String(row.aoi_id),
    aoiName: String(row.aoi_name ?? ""),
    renderedMarkdown: String(row.rendered_markdown ?? ""),
    summary,
    pausedUntil: toDate(row.paused_until),
    quietHours,
    notifyChannels: channels,
    userEmail:
      typeof row.user_email === "string" && row.user_email.length > 0
        ? row.user_email
        : null,
  };
}

function parseChannels(
  raw: unknown,
): Array<
  | { type: "email"; target: string }
  | { type: "webhook"; target: string }
> {
  let arr: unknown = raw;
  if (typeof raw === "string") {
    try {
      arr = JSON.parse(raw);
    } catch {
      return [];
    }
  }
  if (!Array.isArray(arr)) return [];
  const out: Array<
    | { type: "email"; target: string }
    | { type: "webhook"; target: string }
  > = [];
  for (const item of arr) {
    if (
      item &&
      typeof item === "object" &&
      "type" in item &&
      "target" in item &&
      typeof (item as { target: unknown }).target === "string"
    ) {
      const t = (item as { type: unknown }).type;
      const target = (item as { target: string }).target;
      if (t === "email") out.push({ type: "email", target });
      else if (t === "webhook") out.push({ type: "webhook", target });
    }
  }
  return out;
}

function parseQuietHours(
  raw: unknown,
): { tz: string; startHour: number; endHour: number } | null {
  let v: unknown = raw;
  if (typeof v === "string") {
    try {
      v = JSON.parse(v);
    } catch {
      return null;
    }
  }
  if (!v || typeof v !== "object") return null;
  const o = v as { tz?: unknown; startHour?: unknown; endHour?: unknown };
  if (
    typeof o.tz === "string" &&
    typeof o.startHour === "number" &&
    typeof o.endHour === "number"
  ) {
    return { tz: o.tz, startHour: o.startHour, endHour: o.endHour };
  }
  return null;
}

async function findExistingTerminalRow(
  db: AppDb,
  briefId: string,
  channel: string,
  targetHash: string,
): Promise<boolean> {
  const result = (await db.execute(sql`
    SELECT 1 AS one
    FROM "notifications_log"
    WHERE "brief_id" = ${briefId}
      AND "channel" = ${channel}
      AND "target_hash" = ${targetHash}
      AND "status" IN ('sent', 'skipped')
    LIMIT 1
  `)) as unknown as { rows?: Array<{ one: number }> };
  const rows = (result.rows ?? (result as unknown as Array<{ one: number }>)) as Array<{
    one: number;
  }>;
  return rows.length > 0;
}

type PersistArgs = {
  aoiId: string;
  briefId: string;
  channel: string;
  target: string;
  targetHash?: string;
  status: "sent" | "failed" | "skipped" | "config_missing";
  providerMessageId?: string;
  error?: string;
  skipReason?: string;
};

async function persistAttempt(db: AppDb, args: PersistArgs): Promise<void> {
  const targetHash = args.targetHash ?? sha256(args.target);
  await db.execute(sql`
    INSERT INTO "notifications_log" (
      "aoi_id", "brief_id", "channel", "target", "target_hash",
      "status", "provider_message_id", "error", "skip_reason"
    ) VALUES (
      ${args.aoiId},
      ${args.briefId},
      ${args.channel},
      ${args.target},
      ${targetHash},
      ${args.status},
      ${args.providerMessageId ?? null},
      ${args.error ?? null},
      ${args.skipReason ?? null}
    )
    ON CONFLICT DO NOTHING
  `);
}

async function persistSuccess(
  db: AppDb,
  args: {
    aoiId: string;
    briefId: string;
    target: string;
    targetHash: string;
    providerMessageId: string;
    updateLastNotified: boolean;
  },
): Promise<void> {
  await db.transaction(async (tx) => {
    await tx.execute(sql`
      INSERT INTO "notifications_log" (
        "aoi_id", "brief_id", "channel", "target", "target_hash",
        "status", "provider_message_id"
      ) VALUES (
        ${args.aoiId},
        ${args.briefId},
        'email',
        ${args.target},
        ${args.targetHash},
        'sent',
        ${args.providerMessageId}
      )
      ON CONFLICT DO NOTHING
    `);
    if (args.updateLastNotified) {
      await tx.execute(sql`
        UPDATE "aoi_briefs"
        SET "last_notified_at" = COALESCE("last_notified_at", now())
        WHERE "id" = ${args.briefId}
      `);
    }
  });
}

function toDate(v: unknown): Date | null {
  if (v == null) return null;
  if (v instanceof Date) return v;
  if (typeof v === "string" || typeof v === "number") {
    const d = new Date(v);
    return Number.isFinite(d.getTime()) ? d : null;
  }
  return null;
}
