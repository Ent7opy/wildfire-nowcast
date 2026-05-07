/**
 * Stage 7 — side-effect helpers for token actions.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import {
  applySnooze,
  applyPause,
  applyUnsubscribe,
  applyFeedback,
} from "@/lib/notify/actions";
import type { LoadedToken } from "@/lib/notify/action-tokens";
import type { PGlite } from "@electric-sql/pglite";

async function seedAoiWithRules(
  db: AppDb,
  channels: Array<{ type: "email" | "webhook"; target: string }> = [],
): Promise<{ aoiId: string; briefId: string }> {
  const userId = `u-${Math.random().toString(36).slice(2, 8)}`;
  await db.execute(sql`INSERT INTO "users" (id, email) VALUES (${userId}, ${userId + "@x"})`);
  const polygon = JSON.stringify({
    type: "Polygon",
    coordinates: [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
  });
  const aoi = (await db.execute(sql`
    INSERT INTO "aois" (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
    VALUES (${userId}, 'A', ${polygon}, ${polygon},
            ${JSON.stringify({ type: "Point", coordinates: [0.5, 0.5] })},
            '5x5:E000_N00', 100) RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const aoiId = ((aoi.rows ?? aoi) as Array<{ id: string }>)[0].id;
  await db.execute(sql`
    INSERT INTO "aoi_rules" (aoi_id, distance_buffer_km, min_confidence, min_frp_mw, notify_channels)
    VALUES (${aoiId}, 25, 'nominal', 5, ${JSON.stringify(channels)}::jsonb)
    ON CONFLICT ("aoi_id") DO UPDATE SET "notify_channels" = EXCLUDED."notify_channels"
  `);
  const ev = (await db.execute(sql`
    INSERT INTO "aoi_events" (aoi_id, first_seen_at, last_seen_at, nearest_distance_km,
      detection_count, dedupe_hash, status)
    VALUES (${aoiId}, '2026-04-21T00:00:00Z', '2026-04-21T01:00:00Z', 1, 1, 'h', 'new')
    RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const eventId = ((ev.rows ?? ev) as Array<{ id: string }>)[0].id;
  const brief = (await db.execute(sql`
    INSERT INTO "aoi_briefs" (aoi_id, event_id, model, gate_reason, payload, rendered_markdown)
    VALUES (${aoiId}, ${eventId}, 'test', 'multi_pixel',
            ${JSON.stringify({ summary: "x" })}::jsonb, '# brief')
    RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  return {
    aoiId,
    briefId: ((brief.rows ?? brief) as Array<{ id: string }>)[0].id,
  };
}

function makeLoaded(args: {
  aoiId: string;
  briefId: string;
  action: LoadedToken["action"];
  target: string;
}): LoadedToken {
  return {
    token: "tok-" + Math.random(),
    aoiId: args.aoiId,
    briefId: args.briefId,
    action: args.action,
    channel: "email",
    target: args.target,
    expiresAt: new Date(Date.now() + 86400_000),
    redeemedAt: null,
    redeemedValue: null,
  };
}

async function readPaused(db: AppDb, aoiId: string): Promise<Date | null> {
  const r = (await db.execute(sql`
    SELECT "paused_until" FROM "aoi_rules" WHERE "aoi_id" = ${aoiId}
  `)) as unknown as { rows?: Array<{ paused_until: Date | string | null }> };
  const rows = ((r.rows ?? r) as Array<{ paused_until: Date | string | null }>);
  const v = rows[0]?.paused_until ?? null;
  return v == null ? null : v instanceof Date ? v : new Date(v);
}

async function readChannels(db: AppDb, aoiId: string): Promise<unknown> {
  const r = (await db.execute(sql`
    SELECT "notify_channels" FROM "aoi_rules" WHERE "aoi_id" = ${aoiId}
  `)) as unknown as { rows?: Array<{ notify_channels: unknown }> };
  return ((r.rows ?? r) as Array<{ notify_channels: unknown }>)[0]?.notify_channels;
}

describe("notify actions — PGlite", () => {
  let db: AppDb;
  let pglite: PGlite;
  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
  });
  afterEach(async () => {
    await pglite.close();
  });

  it("snooze advances paused_until by 24h", async () => {
    const { aoiId, briefId } = await seedAoiWithRules(db);
    const now = new Date("2026-04-21T00:00:00Z");
    const out = await applySnooze(
      db,
      makeLoaded({ aoiId, briefId, action: "snooze", target: "x@x.org" }),
      now,
    );
    expect(out.pausedUntil.getTime() - now.getTime()).toBe(24 * 3600_000);
  });

  it("snooze preserves a later existing paused_until", async () => {
    const { aoiId, briefId } = await seedAoiWithRules(db);
    const now = new Date("2026-04-21T00:00:00Z");
    const farFuture = new Date(now.getTime() + 7 * 86400_000);
    await db.execute(sql`
      UPDATE "aoi_rules" SET "paused_until" = ${farFuture.toISOString()}
      WHERE "aoi_id" = ${aoiId}
    `);
    const out = await applySnooze(
      db,
      makeLoaded({ aoiId, briefId, action: "snooze", target: "x@x.org" }),
      now,
    );
    expect(out.pausedUntil.toISOString()).toBe(farFuture.toISOString());
  });

  it("pause sets paused_until far in the future", async () => {
    const { aoiId, briefId } = await seedAoiWithRules(db);
    const now = new Date("2026-04-21T00:00:00Z");
    await applyPause(
      db,
      makeLoaded({ aoiId, briefId, action: "pause", target: "x@x.org" }),
      now,
    );
    const stored = await readPaused(db, aoiId);
    expect(stored).not.toBeNull();
    expect(stored!.getTime() - now.getTime()).toBeGreaterThan(50 * 365 * 86400_000);
  });

  it("unsubscribe removes the email channel and leaves others", async () => {
    const { aoiId, briefId } = await seedAoiWithRules(db, [
      { type: "email", target: "remove@x.org" },
      { type: "email", target: "keep@x.org" },
      { type: "webhook", target: "https://hook" },
    ]);
    const out = await applyUnsubscribe(
      db,
      makeLoaded({ aoiId, briefId, action: "unsubscribe", target: "remove@x.org" }),
      new Date(),
    );
    expect(out.autoPaused).toBe(false);
    expect(out.remainingChannels).toHaveLength(2);
    const stored = await readChannels(db, aoiId);
    const parsed = typeof stored === "string" ? JSON.parse(stored) : stored;
    const targets = (parsed as Array<{ target: string }>).map((c) => c.target);
    expect(targets).toEqual(["keep@x.org", "https://hook"]);
  });

  it("unsubscribe auto-pauses when channel list becomes empty", async () => {
    const { aoiId, briefId } = await seedAoiWithRules(db, [
      { type: "email", target: "only@x.org" },
    ]);
    const out = await applyUnsubscribe(
      db,
      makeLoaded({ aoiId, briefId, action: "unsubscribe", target: "only@x.org" }),
      new Date(),
    );
    expect(out.autoPaused).toBe(true);
    const stored = await readPaused(db, aoiId);
    expect(stored).not.toBeNull();
  });

  it("feedback yes inserts row; flip to no updates the same row (unique index)", async () => {
    const { aoiId, briefId } = await seedAoiWithRules(db);
    const loaded = makeLoaded({ aoiId, briefId, action: "feedback", target: "x@x.org" });
    await applyFeedback(db, loaded, "yes", new Date());
    let r = (await db.execute(sql`
      SELECT helpful FROM "brief_feedback" WHERE "brief_id" = ${briefId}
    `)) as unknown as { rows?: Array<{ helpful: boolean }> };
    let rows = ((r.rows ?? r) as Array<{ helpful: boolean }>);
    expect(rows).toHaveLength(1);
    expect(rows[0].helpful).toBe(true);
    await applyFeedback(db, loaded, "no", new Date());
    r = (await db.execute(sql`
      SELECT helpful FROM "brief_feedback" WHERE "brief_id" = ${briefId}
    `)) as unknown as { rows?: Array<{ helpful: boolean }> };
    rows = ((r.rows ?? r) as Array<{ helpful: boolean }>);
    expect(rows).toHaveLength(1);
    expect(rows[0].helpful).toBe(false);
  });

  it("unsubscribe target match is case-sensitive (no false-positive removal)", async () => {
    const { aoiId, briefId } = await seedAoiWithRules(db, [
      { type: "email", target: "Keep@X.org" },
    ]);
    const out = await applyUnsubscribe(
      db,
      makeLoaded({ aoiId, briefId, action: "unsubscribe", target: "keep@x.org" }),
      new Date(),
    );
    expect(out.autoPaused).toBe(false);
    expect(out.remainingChannels).toHaveLength(1);
    expect(out.remainingChannels[0]).toMatchObject({ target: "Keep@X.org" });
  });

  it("unsubscribe leaves a webhook with same target untouched", async () => {
    const { aoiId, briefId } = await seedAoiWithRules(db, [
      { type: "webhook", target: "user@x.org" },
    ]);
    const out = await applyUnsubscribe(
      db,
      makeLoaded({ aoiId, briefId, action: "unsubscribe", target: "user@x.org" }),
      new Date(),
    );
    expect(out.autoPaused).toBe(false);
    expect(out.remainingChannels).toHaveLength(1);
    expect(out.remainingChannels[0]).toMatchObject({ type: "webhook" });
  });

  it("unsubscribe on already-empty channel list still auto-pauses", async () => {
    const { aoiId, briefId } = await seedAoiWithRules(db, []);
    const out = await applyUnsubscribe(
      db,
      makeLoaded({ aoiId, briefId, action: "unsubscribe", target: "any@x.org" }),
      new Date(),
    );
    expect(out.autoPaused).toBe(true);
    expect(out.remainingChannels).toEqual([]);
    const stored = await readPaused(db, aoiId);
    expect(stored).not.toBeNull();
  });

  it("feedback throws when token has no associated brief", async () => {
    const { aoiId, briefId } = await seedAoiWithRules(db);
    const loaded = makeLoaded({ aoiId, briefId, action: "feedback", target: "x@x.org" });
    loaded.briefId = null;
    await expect(applyFeedback(db, loaded, "yes", new Date())).rejects.toThrow(
      /no associated brief/,
    );
  });

  it("rapid yes/no/yes flips converge to last value", async () => {
    const { aoiId, briefId } = await seedAoiWithRules(db);
    const loaded = makeLoaded({ aoiId, briefId, action: "feedback", target: "x@x.org" });
    await applyFeedback(db, loaded, "yes", new Date());
    await applyFeedback(db, loaded, "no", new Date());
    await applyFeedback(db, loaded, "yes", new Date());
    const r = (await db.execute(sql`
      SELECT helpful FROM "brief_feedback" WHERE "brief_id" = ${briefId}
    `)) as unknown as { rows?: Array<{ helpful: boolean }> };
    const rows = ((r.rows ?? r) as Array<{ helpful: boolean }>);
    expect(rows).toHaveLength(1);
    expect(rows[0].helpful).toBe(true);
  });

  it("double-yes is idempotent (no duplicate row)", async () => {
    const { aoiId, briefId } = await seedAoiWithRules(db);
    const loaded = makeLoaded({ aoiId, briefId, action: "feedback", target: "x@x.org" });
    await applyFeedback(db, loaded, "yes", new Date());
    await applyFeedback(db, loaded, "yes", new Date());
    const r = (await db.execute(sql`
      SELECT count(*)::int AS n FROM "brief_feedback" WHERE "brief_id" = ${briefId}
    `)) as unknown as { rows?: Array<{ n: number }> };
    const rows = ((r.rows ?? r) as Array<{ n: number }>);
    expect(Number(rows[0].n)).toBe(1);
  });
});
