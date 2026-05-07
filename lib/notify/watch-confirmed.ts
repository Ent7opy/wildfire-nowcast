/**
 * Stage 9 — watch-confirmed email dispatcher.
 *
 * Single entry point: `dispatchWatchConfirmed`. Sends the one-shot
 * "Now watching {AOI name}. First poll at {UTC time}." email after AOI
 * creation. Uses the Stage 4 `sendEmail` Resend client; persists a
 * `notifications_log` row with `kind='watch_confirmed'` and `brief_id=NULL`.
 *
 * Idempotency: hash on `(aoi_id)` only. A duplicate AOI POST (rare) finds the
 * prior 'sent' row and returns `{ status: "skipped", reason: "duplicate" }`.
 *
 * Two-backend: only non-spatial SQL touched here; identical on Neon+PostGIS
 * and PGlite.
 */
import { createHash } from "node:crypto";
import { sql } from "drizzle-orm";
import type { AppDb } from "@/lib/db/client";
import { sendEmail, type SendResult } from "./resend";
import { renderWatchConfirmedEmail } from "./watch-confirmed-template";

export type WatchConfirmedOutcome =
  | { status: "sent"; providerMessageId: string }
  | {
      status: "skipped";
      reason: "no_recipient" | "no_recipient_pending" | "duplicate";
    }
  | { status: "config_missing" }
  | { status: "failed"; error: string };

export type DispatchWatchConfirmedArgs = {
  aoiId: string;
  userId: string;
  aoiName: string;
  regionBucket: string;
  areaHa: number;
  firstPollAt: Date;
  aoiUrl: string;
  now?: Date;
  sendImpl?: (args: {
    to: string;
    subject: string;
    markdown: string;
  }) => Promise<SendResult>;
};

export async function dispatchWatchConfirmed(
  db: AppDb,
  args: DispatchWatchConfirmedArgs,
): Promise<WatchConfirmedOutcome> {
  const targetHash = sha256(`watch_confirmed:${args.aoiId}`);

  const prior = await findPriorRow(db, args.aoiId, targetHash);
  if (prior) {
    return { status: "skipped", reason: "duplicate" };
  }

  const userEmail = await loadUserEmail(db, args.userId);
  if (!userEmail) {
    await insertRow(db, {
      aoiId: args.aoiId,
      target: "",
      targetHash,
      status: "skipped",
      skipReason: "no_recipient",
    });
    return { status: "skipped", reason: "no_recipient" };
  }
  if (isPendingPlaceholder(userEmail)) {
    await insertRow(db, {
      aoiId: args.aoiId,
      target: userEmail,
      targetHash,
      status: "skipped",
      skipReason: "no_recipient_pending",
    });
    return { status: "skipped", reason: "no_recipient_pending" };
  }

  const { subject, markdown } = renderWatchConfirmedEmail({
    aoiName: args.aoiName,
    regionBucket: args.regionBucket,
    areaHa: args.areaHa,
    firstPollAt: args.firstPollAt,
    aoiUrl: args.aoiUrl,
  });

  const send = args.sendImpl ?? ((a) => sendEmail(a));
  const result = await send({ to: userEmail, subject, markdown });

  if (!result.ok && result.code === "config_missing") {
    await insertRow(db, {
      aoiId: args.aoiId,
      target: userEmail,
      targetHash,
      status: "config_missing",
    });
    return { status: "config_missing" };
  }

  if (!result.ok) {
    const error = `${result.code}: ${result.message}`;
    await insertRow(db, {
      aoiId: args.aoiId,
      target: userEmail,
      targetHash,
      status: "failed",
      error: truncate(error, 500),
    });
    return { status: "failed", error };
  }

  await insertRow(db, {
    aoiId: args.aoiId,
    target: userEmail,
    targetHash,
    status: "sent",
    providerMessageId: result.providerMessageId,
  });
  return { status: "sent", providerMessageId: result.providerMessageId };
}

// ---------------------------------------------------------------------------
// Internals

type InsertArgs = {
  aoiId: string;
  target: string;
  targetHash: string;
  status: "sent" | "failed" | "skipped" | "config_missing";
  providerMessageId?: string;
  error?: string;
  skipReason?: string;
};

async function insertRow(db: AppDb, args: InsertArgs): Promise<void> {
  await db.execute(sql`
    INSERT INTO "notifications_log" (
      "aoi_id", "brief_id", "kind", "channel", "target", "target_hash",
      "status", "provider_message_id", "error", "skip_reason"
    ) VALUES (
      ${args.aoiId},
      NULL,
      'watch_confirmed',
      'email',
      ${args.target},
      ${args.targetHash},
      ${args.status},
      ${args.providerMessageId ?? null},
      ${args.error ?? null},
      ${args.skipReason ?? null}
    )
  `);
}

async function findPriorRow(
  db: AppDb,
  aoiId: string,
  targetHash: string,
): Promise<boolean> {
  const result = (await db.execute(sql`
    SELECT 1 AS one FROM "notifications_log"
    WHERE "aoi_id" = ${aoiId}
      AND "kind" = 'watch_confirmed'
      AND "target_hash" = ${targetHash}
      AND "status" IN ('sent', 'skipped', 'config_missing')
    LIMIT 1
  `)) as unknown as { rows?: Array<{ one: number }> };
  const rows = (result.rows ?? (result as unknown as Array<{ one: number }>)) as Array<{
    one: number;
  }>;
  return rows.length > 0;
}

async function loadUserEmail(db: AppDb, userId: string): Promise<string | null> {
  const result = (await db.execute(sql`
    SELECT "email" FROM "users" WHERE "id" = ${userId} LIMIT 1
  `)) as unknown as { rows?: Array<{ email: string | null }> };
  const rows = (result.rows ?? (result as unknown as Array<{ email: string | null }>)) as Array<{
    email: string | null;
  }>;
  const email = rows[0]?.email;
  return typeof email === "string" && email.length > 0 ? email : null;
}

function isPendingPlaceholder(email: string): boolean {
  return /@pending\.invalid$/.test(email);
}

function sha256(s: string): string {
  return createHash("sha256").update(s).digest("hex");
}

function truncate(s: string, max: number): string {
  return s.length <= max ? s : s.slice(0, max);
}

export function absoluteAoiUrl(aoiId: string): string {
  const host = process.env.NEXT_PUBLIC_APP_URL?.replace(/\/$/, "") ?? "http://localhost:3000";
  return `${host}/dashboard/aoi/${aoiId}`;
}
