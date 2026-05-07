/**
 * Stage 4 notification dispatcher. Single entry: `dispatchBrief`.
 * See `pm/briefs/18-stage4-notification-dispatch.md` §Dispatcher.
 *
 * Two-backend: every SQL touched is non-spatial and runs identically on
 * Neon+PostGIS and PGlite.
 */
import { createHash } from "node:crypto";
import { sql } from "drizzle-orm";
import type { AppDb } from "@/lib/db/client";
import { sendEmail, type SendResult } from "./resend";
import { mintActionToken } from "./action-tokens";
import { appendFooter, type FooterUrls } from "./footer";
import { notifyActionUrl } from "@/lib/share/url";

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

type Channel =
  | { type: "email"; target: string }
  | { type: "webhook"; target: string };

type LoadedBrief = {
  briefId: string;
  aoiId: string;
  aoiName: string;
  renderedMarkdown: string;
  summary: string;
  pausedUntil: Date | null;
  quietHours: { tz: string; startHour: number; endHour: number } | null;
  notifyChannels: Channel[];
  userEmail: string | null;
};

export async function dispatchBrief(
  db: AppDb,
  briefId: string,
  deps: DispatchDeps = {},
): Promise<DispatchOutcome> {
  const loaded = await loadBrief(db, briefId);
  if (!loaded) return { briefId, attempts: [] };

  const now = deps.now ?? new Date();
  const send = deps.send ?? ((a) => sendEmail(a));
  const channels = resolveChannels(loaded);
  const attempts: DispatchAttempt[] = [];

  if (channels.length === 0) {
    const reason = isPendingPlaceholder(loaded.userEmail)
      ? "no_recipient_pending"
      : "no_recipient";
    const target = loaded.userEmail ?? "";
    await insertNotificationRow(db, {
      aoiId: loaded.aoiId,
      briefId,
      channel: "email",
      target,
      status: "skipped",
      skipReason: reason,
    });
    attempts.push({ status: "skipped", channel: "email", target, reason });
    return { briefId, attempts };
  }

  let firstSuccessRecorded = false;
  for (const ch of channels) {
    const target = ch.target;
    const targetHash = sha256(target);
    const base = { aoiId: loaded.aoiId, briefId, channel: ch.type, target, targetHash };

    const skip = async (reason: string): Promise<void> => {
      await insertNotificationRow(db, { ...base, status: "skipped", skipReason: reason });
      attempts.push({ status: "skipped", channel: ch.type, target, reason });
    };

    if (await findExistingTerminalRow(db, briefId, ch.type, targetHash)) {
      attempts.push({ status: "skipped", channel: ch.type, target, reason: "duplicate" });
      continue;
    }

    if (ch.type === "webhook") {
      await skip("channel_not_implemented");
      continue;
    }

    if (loaded.pausedUntil && loaded.pausedUntil.getTime() > now.getTime()) {
      await skip("paused");
      continue;
    }

    // TODO Stage 6: hold + release as a morning digest at the top of the
    // quiet window per US-2 acceptance #3. Stage 4 is skip-only.
    if (loaded.quietHours && inQuietHours(now, loaded.quietHours)) {
      await skip("quiet_hours");
      continue;
    }

    // Mint a fresh quartet of action tokens for THIS outbound email — never
    // reused across emails so forwarding cannot grant the recipient
    // permission to mutate the AOI.
    const footerUrls = await mintFooterUrls(db, {
      aoiId: loaded.aoiId,
      briefId,
      target,
      now,
    });
    const result = await send({
      to: target,
      subject: truncate(loaded.summary, 90),
      markdown: appendFooter(loaded.renderedMarkdown, footerUrls),
    });

    if (!result.ok && result.code === "config_missing") {
      await insertNotificationRow(db, { ...base, status: "config_missing" });
      attempts.push({ status: "config_missing", channel: "email", target });
      continue;
    }

    if (!result.ok) {
      const error = `${result.code}: ${result.message}`;
      await insertNotificationRow(db, {
        ...base,
        status: "failed",
        error: truncate(error, 500),
      });
      attempts.push({ status: "failed", channel: "email", target, error });
      continue;
    }

    await persistSuccess(db, {
      aoiId: loaded.aoiId,
      briefId,
      target,
      targetHash,
      providerMessageId: result.providerMessageId,
      updateLastNotified: !firstSuccessRecorded,
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

const FOOTER_ACTIONS = ["snooze", "pause", "unsubscribe", "feedback"] as const;

async function mintFooterUrls(
  db: AppDb,
  args: { aoiId: string; briefId: string; target: string; now: Date },
): Promise<FooterUrls> {
  const tokens = {} as Record<(typeof FOOTER_ACTIONS)[number], string>;
  for (const action of FOOTER_ACTIONS) {
    const { token } = await mintActionToken(db, {
      aoiId: args.aoiId,
      briefId: args.briefId,
      action,
      channel: "email",
      target: args.target,
      now: args.now,
    });
    tokens[action] = token;
  }
  return {
    snoozeUrl: notifyActionUrl("snooze", tokens.snooze),
    pauseUrl: notifyActionUrl("pause", tokens.pause),
    unsubscribeUrl: notifyActionUrl("unsubscribe", tokens.unsubscribe),
    feedbackYesUrl: notifyActionUrl("feedback", tokens.feedback, { v: "yes" }),
    feedbackNoUrl: notifyActionUrl("feedback", tokens.feedback, { v: "no" }),
  };
}

function resolveChannels(loaded: LoadedBrief): Channel[] {
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

function truncate(s: string, max: number): string {
  return s.length <= max ? s : s.slice(0, max);
}

function sha256(s: string): string {
  return createHash("sha256").update(s).digest("hex");
}

// Wraparound (e.g. start=22, end=7) handled via OR.
export function inQuietHours(
  now: Date,
  qh: { tz: string; startHour: number; endHour: number },
): boolean {
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: qh.tz,
    hour: "numeric",
    hour12: false,
  });
  const hourPart = fmt.formatToParts(now).find((p) => p.type === "hour");
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

function rowsOf<T>(result: unknown): T[] {
  const r = result as { rows?: T[] } | T[];
  if (Array.isArray(r)) return r;
  return r.rows ?? [];
}

async function loadBrief(db: AppDb, briefId: string): Promise<LoadedBrief | null> {
  const result = await db.execute(sql`
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
  `);
  const row = rowsOf<Record<string, unknown>>(result)[0];
  if (!row) return null;

  return {
    briefId: String(row.brief_id),
    aoiId: String(row.aoi_id),
    aoiName: String(row.aoi_name ?? ""),
    renderedMarkdown: String(row.rendered_markdown ?? ""),
    summary: parseSummary(row.payload),
    pausedUntil: toDate(row.paused_until),
    quietHours: parseQuietHours(row.quiet_hours),
    notifyChannels: parseChannels(row.notify_channels),
    userEmail:
      typeof row.user_email === "string" && row.user_email.length > 0
        ? row.user_email
        : null,
  };
}

function parseJsonish(raw: unknown): unknown {
  if (typeof raw !== "string") return raw;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function parseSummary(payload: unknown): string {
  const v = parseJsonish(payload);
  if (v && typeof v === "object" && "summary" in v) {
    const s = (v as { summary: unknown }).summary;
    if (typeof s === "string") return s;
  }
  return "";
}

function parseChannels(raw: unknown): Channel[] {
  const arr = parseJsonish(raw);
  if (!Array.isArray(arr)) return [];
  const out: Channel[] = [];
  for (const item of arr) {
    if (!item || typeof item !== "object") continue;
    const { type, target } = item as { type?: unknown; target?: unknown };
    if (typeof target !== "string") continue;
    if (type === "email" || type === "webhook") out.push({ type, target });
  }
  return out;
}

function parseQuietHours(
  raw: unknown,
): { tz: string; startHour: number; endHour: number } | null {
  const v = parseJsonish(raw);
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
  const result = await db.execute(sql`
    SELECT 1 AS one
    FROM "notifications_log"
    WHERE "brief_id" = ${briefId}
      AND "channel" = ${channel}
      AND "target_hash" = ${targetHash}
      AND "status" IN ('sent', 'skipped')
    LIMIT 1
  `);
  return rowsOf(result).length > 0;
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
    await insertNotificationRow(tx, {
      aoiId: args.aoiId,
      briefId: args.briefId,
      channel: "email",
      target: args.target,
      targetHash: args.targetHash,
      status: "sent",
      providerMessageId: args.providerMessageId,
    });
    if (args.updateLastNotified) {
      await tx.execute(sql`
        UPDATE "aoi_briefs"
        SET "last_notified_at" = COALESCE("last_notified_at", now())
        WHERE "id" = ${args.briefId}
      `);
    }
  });
}

async function insertNotificationRow(
  exec: { execute: AppDb["execute"] },
  args: PersistArgs,
): Promise<void> {
  const targetHash = args.targetHash ?? sha256(args.target);
  await exec.execute(sql`
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

function toDate(v: unknown): Date | null {
  if (v == null) return null;
  if (v instanceof Date) return v;
  if (typeof v === "string" || typeof v === "number") {
    const d = new Date(v);
    return Number.isFinite(d.getTime()) ? d : null;
  }
  return null;
}
