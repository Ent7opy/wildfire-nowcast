/**
 * Stage 7 — side effects of redeeming a notify-action token.
 *
 * Each function is intentionally narrow: it takes the loaded token row and
 * applies its single effect. Idempotency is handled by the caller (the route
 * handler) reading the token's `redeemedAt` flag.
 *
 * SQL is non-spatial; runs identically on Neon+PostGIS and PGlite.
 */
import { sql } from "drizzle-orm";
import type { AppDb } from "@/lib/db/client";
import { decodeRows } from "@/lib/db/decode-rows";
import type { LoadedToken } from "./action-tokens";

const SNOOZE_HOURS = 24;
const PAUSE_INDEFINITE_YEARS = 100;

type ChannelEntry =
  | { type: "email"; target: string }
  | { type: "webhook"; target: string };

async function readRules(
  db: AppDb,
  aoiId: string,
): Promise<{ pausedUntil: Date | null; channels: ChannelEntry[] } | null> {
  const result = await db.execute(sql`
    SELECT "paused_until", "notify_channels"
    FROM "aoi_rules"
    WHERE "aoi_id" = ${aoiId}
    LIMIT 1
  `);
  const rows = decodeRows<{
    paused_until: Date | string | null;
    notify_channels: unknown;
  }>(result);
  const row = rows[0];
  if (!row) return null;
  const pausedUntil =
    row.paused_until == null
      ? null
      : row.paused_until instanceof Date
        ? row.paused_until
        : new Date(row.paused_until);
  let channels: ChannelEntry[] = [];
  let raw: unknown = row.notify_channels;
  if (typeof raw === "string") {
    try {
      raw = JSON.parse(raw);
    } catch {
      raw = [];
    }
  }
  if (Array.isArray(raw)) {
    channels = raw.filter(
      (c): c is ChannelEntry =>
        !!c &&
        typeof c === "object" &&
        "type" in c &&
        "target" in c &&
        typeof (c as { target: unknown }).target === "string" &&
        ((c as { type: unknown }).type === "email" ||
          (c as { type: unknown }).type === "webhook"),
    );
  }
  return { pausedUntil, channels };
}

export async function applySnooze(
  db: AppDb,
  loaded: LoadedToken,
  now: Date,
): Promise<{ pausedUntil: Date }> {
  const current = await readRules(db, loaded.aoiId);
  const candidate = new Date(now.getTime() + SNOOZE_HOURS * 3600_000);
  const next =
    current?.pausedUntil && current.pausedUntil.getTime() > candidate.getTime()
      ? current.pausedUntil
      : candidate;
  await ensureRulesRow(db, loaded.aoiId);
  await db.execute(sql`
    UPDATE "aoi_rules"
    SET "paused_until" = ${next.toISOString()}, "updated_at" = ${now.toISOString()}
    WHERE "aoi_id" = ${loaded.aoiId}
  `);
  return { pausedUntil: next };
}

export async function applyPause(
  db: AppDb,
  loaded: LoadedToken,
  now: Date,
): Promise<{ pausedUntil: Date }> {
  const indefinite = new Date(now.getTime());
  indefinite.setUTCFullYear(indefinite.getUTCFullYear() + PAUSE_INDEFINITE_YEARS);
  await ensureRulesRow(db, loaded.aoiId);
  await db.execute(sql`
    UPDATE "aoi_rules"
    SET "paused_until" = ${indefinite.toISOString()}, "updated_at" = ${now.toISOString()}
    WHERE "aoi_id" = ${loaded.aoiId}
  `);
  return { pausedUntil: indefinite };
}

export type UnsubscribeOutcome = {
  remainingChannels: ChannelEntry[];
  autoPaused: boolean;
};

export async function applyUnsubscribe(
  db: AppDb,
  loaded: LoadedToken,
  now: Date,
): Promise<UnsubscribeOutcome> {
  const current = await readRules(db, loaded.aoiId);
  const channels = current?.channels ?? [];
  const remaining = channels.filter(
    (c) => !(c.type === "email" && c.target === loaded.target),
  );
  await ensureRulesRow(db, loaded.aoiId);
  let autoPaused = false;
  let pausedUntilSet: string | null = null;
  if (remaining.length === 0) {
    const indefinite = new Date(now.getTime());
    indefinite.setUTCFullYear(indefinite.getUTCFullYear() + PAUSE_INDEFINITE_YEARS);
    pausedUntilSet = indefinite.toISOString();
    autoPaused = true;
  }
  await db.execute(sql`
    UPDATE "aoi_rules"
    SET
      "notify_channels" = ${JSON.stringify(remaining)}::jsonb,
      "paused_until"    = COALESCE(${pausedUntilSet}, "paused_until"),
      "updated_at"      = ${now.toISOString()}
    WHERE "aoi_id" = ${loaded.aoiId}
  `);
  return { remainingChannels: remaining, autoPaused };
}

export type FeedbackValue = "yes" | "no";

export async function applyFeedback(
  db: AppDb,
  loaded: LoadedToken,
  value: FeedbackValue,
  now: Date,
): Promise<{ helpful: boolean }> {
  if (!loaded.briefId) {
    throw new Error("applyFeedback: token has no associated brief");
  }
  const helpful = value === "yes";
  await db.execute(sql`
    INSERT INTO "brief_feedback" ("brief_id", "helpful", "recipient_token", "created_at")
    VALUES (${loaded.briefId}, ${helpful}, ${loaded.token}, ${now.toISOString()})
    ON CONFLICT ("brief_id", "recipient_token")
    DO UPDATE SET "helpful" = EXCLUDED."helpful"
  `);
  return { helpful };
}

/**
 * Ensures an `aoi_rules` row exists for this AOI before an UPDATE — defends
 * against the corner case where Stage 1's auto-insert was missed (legacy
 * rows from migration windows). Idempotent.
 */
async function ensureRulesRow(db: AppDb, aoiId: string): Promise<void> {
  await db.execute(sql`
    INSERT INTO "aoi_rules" ("aoi_id")
    VALUES (${aoiId})
    ON CONFLICT ("aoi_id") DO NOTHING
  `);
}
