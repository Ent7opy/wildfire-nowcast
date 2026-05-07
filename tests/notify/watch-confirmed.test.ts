/**
 * Stage 9 — watch-confirmed dispatcher unit tests on PGlite.
 *
 * Mirrors the structure of `tests/notify/dispatch.test.ts`. Covers:
 *   - happy path → row inserted, kind='watch_confirmed', status='sent'
 *   - duplicate call → second call returns skipped/duplicate, no second row
 *   - sendImpl returns config_missing → row inserted with status='config_missing'
 *   - JIT @pending.invalid email → skipped/no_recipient_pending, send not called
 *   - missing user → skipped/no_recipient
 *   - sendImpl returns provider_error → row inserted status='failed'
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import type { PGlite } from "@electric-sql/pglite";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import { dispatchWatchConfirmed } from "@/lib/notify/watch-confirmed";
import type { SendResult } from "@/lib/notify/resend";

async function seedAoi(
  db: AppDb,
  opts: { userEmail?: string; userId?: string } = {},
): Promise<{ aoiId: string; userId: string }> {
  const userId = opts.userId ?? `stub-user-${Math.random().toString(36).slice(2, 8)}`;
  const userEmail = opts.userEmail ?? "owner@example.org";
  await db.execute(sql`
    INSERT INTO "users" (id, email) VALUES (${userId}, ${userEmail})
  `);
  const polygon = JSON.stringify({
    type: "Polygon",
    coordinates: [
      [[-122.7, 38.4], [-122.6, 38.4], [-122.6, 38.5], [-122.7, 38.5], [-122.7, 38.4]],
    ],
  });
  const aoiRes = (await db.execute(sql`
    INSERT INTO "aois" (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
    VALUES (
      ${userId}, 'Spring Creek Preserve', ${polygon}, ${polygon},
      ${JSON.stringify({ type: "Point", coordinates: [-122.65, 38.45] })},
      '5x5:W125_N35', 850.5
    ) RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const rows = (aoiRes.rows ?? (aoiRes as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  return { aoiId: rows[0].id, userId };
}

async function readRows(
  db: AppDb,
  aoiId: string,
): Promise<Array<Record<string, unknown>>> {
  const r = (await db.execute(sql`
    SELECT * FROM "notifications_log"
    WHERE "aoi_id" = ${aoiId} AND "kind" = 'watch_confirmed'
    ORDER BY "sent_at" ASC
  `)) as unknown as { rows?: Array<Record<string, unknown>> };
  return (r.rows ?? (r as unknown as Array<Record<string, unknown>>)) as Array<Record<string, unknown>>;
}

const baseArgs = (aoiId: string, userId: string) => ({
  aoiId,
  userId,
  aoiName: "Spring Creek Preserve",
  regionBucket: "5x5:W125_N35",
  areaHa: 850.5,
  firstPollAt: new Date("2026-05-07T14:30:00Z"),
  aoiUrl: "http://localhost:3000/dashboard/aoi/abc",
});

describe("dispatchWatchConfirmed — PGlite", () => {
  let db: AppDb;
  let pglite: PGlite;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
  });

  afterEach(async () => {
    await pglite.close();
  });

  it("happy path — sends and inserts row with kind='watch_confirmed'", async () => {
    const { aoiId, userId } = await seedAoi(db);
    let captured: { to: string; subject: string; markdown: string } | null = null;
    const sendImpl = async (a: {
      to: string;
      subject: string;
      markdown: string;
    }): Promise<SendResult> => {
      captured = a;
      return { ok: true, providerMessageId: "resend-stage9-1", latencyMs: 12 };
    };
    const outcome = await dispatchWatchConfirmed(db, {
      ...baseArgs(aoiId, userId),
      sendImpl,
    });
    expect(outcome.status).toBe("sent");
    if (outcome.status === "sent") {
      expect(outcome.providerMessageId).toBe("resend-stage9-1");
    }
    expect(captured).not.toBeNull();
    expect(captured!.to).toBe("owner@example.org");
    expect(captured!.subject).toContain("Now watching Spring Creek Preserve");
    expect(captured!.markdown).toContain("Spring Creek Preserve");
    expect(captured!.markdown).toContain("2026-05-07 14:30 UTC");

    const rows = await readRows(db, aoiId);
    expect(rows).toHaveLength(1);
    expect(rows[0].status).toBe("sent");
    expect(rows[0].kind).toBe("watch_confirmed");
    expect(rows[0].brief_id).toBeNull();
    expect(rows[0].provider_message_id).toBe("resend-stage9-1");
  });

  it("duplicate call → second invocation is skipped/duplicate, no second row", async () => {
    const { aoiId, userId } = await seedAoi(db);
    const sendImpl = async (): Promise<SendResult> => ({
      ok: true,
      providerMessageId: "x",
      latencyMs: 1,
    });
    await dispatchWatchConfirmed(db, { ...baseArgs(aoiId, userId), sendImpl });
    const second = await dispatchWatchConfirmed(db, {
      ...baseArgs(aoiId, userId),
      sendImpl,
    });
    expect(second.status).toBe("skipped");
    if (second.status === "skipped") {
      expect(second.reason).toBe("duplicate");
    }
    const rows = await readRows(db, aoiId);
    expect(rows).toHaveLength(1);
  });

  it("config_missing → row inserted with status='config_missing'", async () => {
    const { aoiId, userId } = await seedAoi(db);
    const sendImpl = async (): Promise<SendResult> => ({
      ok: false,
      code: "config_missing",
      message: "RESEND_API_KEY not set",
      latencyMs: 0,
    });
    const outcome = await dispatchWatchConfirmed(db, {
      ...baseArgs(aoiId, userId),
      sendImpl,
    });
    expect(outcome.status).toBe("config_missing");
    const rows = await readRows(db, aoiId);
    expect(rows).toHaveLength(1);
    expect(rows[0].status).toBe("config_missing");
  });

  it("@pending.invalid email → skipped/no_recipient_pending, send not called", async () => {
    const { aoiId, userId } = await seedAoi(db, {
      userEmail: "user_xyz@pending.invalid",
    });
    let called = false;
    const sendImpl = async (): Promise<SendResult> => {
      called = true;
      return { ok: true, providerMessageId: "x", latencyMs: 1 };
    };
    const outcome = await dispatchWatchConfirmed(db, {
      ...baseArgs(aoiId, userId),
      sendImpl,
    });
    expect(called).toBe(false);
    expect(outcome.status).toBe("skipped");
    if (outcome.status === "skipped") {
      expect(outcome.reason).toBe("no_recipient_pending");
    }
    const rows = await readRows(db, aoiId);
    expect(rows[0].skip_reason).toBe("no_recipient_pending");
  });

  it("missing user → skipped/no_recipient", async () => {
    const { aoiId } = await seedAoi(db);
    const outcome = await dispatchWatchConfirmed(db, {
      ...baseArgs(aoiId, "nonexistent-user"),
    });
    expect(outcome.status).toBe("skipped");
    if (outcome.status === "skipped") {
      expect(outcome.reason).toBe("no_recipient");
    }
  });

  it("provider_error → row written status='failed'; dispatcher does not throw", async () => {
    const { aoiId, userId } = await seedAoi(db);
    const sendImpl = async (): Promise<SendResult> => ({
      ok: false,
      code: "provider_error",
      message: "boom",
      latencyMs: 0,
    });
    const outcome = await dispatchWatchConfirmed(db, {
      ...baseArgs(aoiId, userId),
      sendImpl,
    });
    expect(outcome.status).toBe("failed");
    const rows = await readRows(db, aoiId);
    expect(rows[0].status).toBe("failed");
    expect(String(rows[0].error)).toContain("provider_error");
  });
});
