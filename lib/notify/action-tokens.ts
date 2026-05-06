/**
 * Stage 7 — signed-token plumbing for email actions
 * (snooze / pause / unsubscribe / feedback).
 *
 * The token is a 32-byte hex string (256 bits of crypto randomness). It IS
 * the auth — there is no Clerk session on the redemption path, because the
 * recipient clicks the link from their inbox.
 *
 * Threat model is the same as the existing `lib/share/token.ts`: the URL
 * lands only in the recipient's mailbox / mail-scanner; bot prefetching of
 * the bearer secret is not a concern. Idempotency on redemption keeps a
 * scanner-induced click recoverable (the user just clicks again).
 */
import { randomBytes } from "node:crypto";
import { sql } from "drizzle-orm";
import type { AppDb } from "@/lib/db/client";

export type ActionKind = "snooze" | "pause" | "unsubscribe" | "feedback";

const TTL_DAYS: Record<ActionKind, number> = {
  snooze: 30,
  pause: 30,
  unsubscribe: 30,
  feedback: 90,
};

let testMint: (() => string) | null = null;

export function _setMintActionTokenForTest(fn: (() => string) | null): void {
  testMint = fn;
}

export function mintActionTokenString(): string {
  if (testMint) return testMint();
  return randomBytes(32).toString("hex");
}

export type MintArgs = {
  aoiId: string;
  briefId: string | null;
  action: ActionKind;
  channel: string;
  target: string;
  now?: Date;
};

export type MintedToken = {
  token: string;
  action: ActionKind;
  expiresAt: Date;
};

export async function mintActionToken(
  db: AppDb,
  args: MintArgs,
): Promise<MintedToken> {
  const now = args.now ?? new Date();
  const expiresAt = new Date(now.getTime() + TTL_DAYS[args.action] * 86400_000);
  const token = mintActionTokenString();
  await db.execute(sql`
    INSERT INTO "notify_action_tokens" (
      "token", "aoi_id", "brief_id", "action", "channel",
      "target", "expires_at", "created_at"
    ) VALUES (
      ${token}, ${args.aoiId}, ${args.briefId}, ${args.action}, ${args.channel},
      ${args.target}, ${expiresAt.toISOString()}, ${now.toISOString()}
    )
  `);
  return { token, action: args.action, expiresAt };
}

export type LoadedToken = {
  token: string;
  aoiId: string;
  briefId: string | null;
  action: ActionKind;
  channel: string;
  target: string;
  expiresAt: Date;
  redeemedAt: Date | null;
  redeemedValue: string | null;
};

function toDate(v: unknown): Date | null {
  if (v == null) return null;
  if (v instanceof Date) return v;
  if (typeof v === "string" || typeof v === "number") {
    const d = new Date(v);
    return Number.isFinite(d.getTime()) ? d : null;
  }
  return null;
}

export async function loadActionToken(
  db: AppDb,
  token: string,
): Promise<LoadedToken | null> {
  if (!token) return null;
  const result = (await db.execute(sql`
    SELECT
      "token", "aoi_id", "brief_id", "action", "channel", "target",
      "expires_at", "redeemed_at", "redeemed_value"
    FROM "notify_action_tokens"
    WHERE "token" = ${token}
    LIMIT 1
  `)) as unknown as { rows?: Array<Record<string, unknown>> };
  const rows = (result.rows ??
    (result as unknown as Array<Record<string, unknown>>)) as Array<
    Record<string, unknown>
  >;
  const row = rows[0];
  if (!row) return null;
  const expiresAt = toDate(row.expires_at);
  if (!expiresAt) return null;
  return {
    token: String(row.token),
    aoiId: String(row.aoi_id),
    briefId: row.brief_id == null ? null : String(row.brief_id),
    action: String(row.action) as ActionKind,
    channel: String(row.channel),
    target: String(row.target),
    expiresAt,
    redeemedAt: toDate(row.redeemed_at),
    redeemedValue:
      row.redeemed_value == null ? null : String(row.redeemed_value),
  };
}

export type RedeemOutcome =
  | { ok: true; first: boolean; loaded: LoadedToken }
  | { ok: false; reason: "not_found" | "expired" | "wrong_action" };

/**
 * Mark the token as redeemed (idempotent). Returns the loaded row plus a
 * `first` flag so the caller can choose whether to apply side effects again
 * (for `feedback` we DO want the second click to flip the row; for snooze /
 * pause / unsubscribe the first redemption is the durable one).
 */
export async function redeemActionToken(
  db: AppDb,
  args: { token: string; expectedAction: ActionKind; redeemedValue?: string; now?: Date },
): Promise<RedeemOutcome> {
  const loaded = await loadActionToken(db, args.token);
  if (!loaded) return { ok: false, reason: "not_found" };
  const now = args.now ?? new Date();
  if (loaded.expiresAt.getTime() <= now.getTime()) {
    return { ok: false, reason: "expired" };
  }
  if (loaded.action !== args.expectedAction) {
    return { ok: false, reason: "wrong_action" };
  }
  const first = loaded.redeemedAt == null;
  if (first) {
    await db.execute(sql`
      UPDATE "notify_action_tokens"
      SET "redeemed_at" = ${now.toISOString()},
          "redeemed_value" = ${args.redeemedValue ?? null}
      WHERE "token" = ${args.token}
    `);
    return {
      ok: true,
      first: true,
      loaded: {
        ...loaded,
        redeemedAt: now,
        redeemedValue: args.redeemedValue ?? null,
      },
    };
  }
  // Re-redemption with a new value (e.g. feedback yes → no flip): update value
  // but keep the original redeemed_at so observers see "first redemption".
  if (
    args.redeemedValue !== undefined &&
    args.redeemedValue !== loaded.redeemedValue
  ) {
    await db.execute(sql`
      UPDATE "notify_action_tokens"
      SET "redeemed_value" = ${args.redeemedValue}
      WHERE "token" = ${args.token}
    `);
    return {
      ok: true,
      first: false,
      loaded: { ...loaded, redeemedValue: args.redeemedValue },
    };
  }
  return { ok: true, first: false, loaded };
}
