/**
 * Stage 7 — notify_action_tokens mint + redeem unit tests on PGlite.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import {
  loadActionToken,
  mintActionToken,
  redeemActionToken,
} from "@/lib/notify/action-tokens";
import type { PGlite } from "@electric-sql/pglite";

async function seedAoi(db: AppDb): Promise<{ aoiId: string; briefId: string }> {
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
            '5x5:E000_N00', 100)
    RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const aoiRows = (aoi.rows ?? (aoi as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  const aoiId = aoiRows[0].id;
  const ev = (await db.execute(sql`
    INSERT INTO "aoi_events" (aoi_id, first_seen_at, last_seen_at, nearest_distance_km,
      detection_count, dedupe_hash, status)
    VALUES (${aoiId}, '2026-04-21T00:00:00Z', '2026-04-21T01:00:00Z', 1, 1, 'h', 'new')
    RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const evRows = (ev.rows ?? (ev as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  const eventId = evRows[0].id;
  const brief = (await db.execute(sql`
    INSERT INTO "aoi_briefs" (aoi_id, event_id, model, gate_reason, payload, rendered_markdown)
    VALUES (${aoiId}, ${eventId}, 'test', 'multi_pixel',
            ${JSON.stringify({ summary: "x" })}::jsonb, '# brief')
    RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const briefRows = (brief.rows ?? (brief as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  return { aoiId, briefId: briefRows[0].id };
}

describe("notify action tokens — PGlite", () => {
  let db: AppDb;
  let pglite: PGlite;
  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
  });
  afterEach(async () => {
    await pglite.close();
  });

  it("mint inserts a row with the right shape and 30-day expiry for snooze", async () => {
    const { aoiId, briefId } = await seedAoi(db);
    const now = new Date("2026-04-21T00:00:00Z");
    const minted = await mintActionToken(db, {
      aoiId,
      briefId,
      action: "snooze",
      channel: "email",
      target: "alice@example.org",
      now,
    });
    expect(minted.token).toMatch(/^[0-9a-f]{64}$/);
    expect(minted.action).toBe("snooze");
    expect(minted.expiresAt.getTime() - now.getTime()).toBe(30 * 86400_000);
    const loaded = await loadActionToken(db, minted.token);
    expect(loaded?.aoiId).toBe(aoiId);
    expect(loaded?.briefId).toBe(briefId);
    expect(loaded?.target).toBe("alice@example.org");
    expect(loaded?.redeemedAt).toBeNull();
  });

  it("feedback gets a 90-day expiry", async () => {
    const { aoiId, briefId } = await seedAoi(db);
    const now = new Date("2026-04-21T00:00:00Z");
    const minted = await mintActionToken(db, {
      aoiId,
      briefId,
      action: "feedback",
      channel: "email",
      target: "x@x.org",
      now,
    });
    expect(minted.expiresAt.getTime() - now.getTime()).toBe(90 * 86400_000);
  });

  it("redeem marks first redemption; second redemption returns first=false", async () => {
    const { aoiId, briefId } = await seedAoi(db);
    const now = new Date();
    const minted = await mintActionToken(db, {
      aoiId, briefId, action: "snooze", channel: "email", target: "x@x.org", now,
    });
    const r1 = await redeemActionToken(db, { token: minted.token, expectedAction: "snooze" });
    expect(r1.ok).toBe(true);
    if (r1.ok) expect(r1.first).toBe(true);
    const r2 = await redeemActionToken(db, { token: minted.token, expectedAction: "snooze" });
    expect(r2.ok).toBe(true);
    if (r2.ok) expect(r2.first).toBe(false);
  });

  it("expired token rejected", async () => {
    const { aoiId, briefId } = await seedAoi(db);
    const past = new Date("2020-01-01T00:00:00Z");
    const minted = await mintActionToken(db, {
      aoiId, briefId, action: "pause", channel: "email", target: "x@x.org", now: past,
    });
    const r = await redeemActionToken(db, { token: minted.token, expectedAction: "pause" });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe("expired");
  });

  it("wrong-action token rejected", async () => {
    const { aoiId, briefId } = await seedAoi(db);
    const minted = await mintActionToken(db, {
      aoiId, briefId, action: "snooze", channel: "email", target: "x@x.org",
    });
    const r = await redeemActionToken(db, { token: minted.token, expectedAction: "pause" });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe("wrong_action");
  });

  it("unknown token returns not_found", async () => {
    const r = await redeemActionToken(db, { token: "doesnotexist", expectedAction: "snooze" });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.reason).toBe("not_found");
  });

  it("feedback flip: yes then no updates redeemed_value", async () => {
    const { aoiId, briefId } = await seedAoi(db);
    const minted = await mintActionToken(db, {
      aoiId, briefId, action: "feedback", channel: "email", target: "x@x.org",
    });
    await redeemActionToken(db, {
      token: minted.token, expectedAction: "feedback", redeemedValue: "yes",
    });
    const loaded1 = await loadActionToken(db, minted.token);
    expect(loaded1?.redeemedValue).toBe("yes");
    await redeemActionToken(db, {
      token: minted.token, expectedAction: "feedback", redeemedValue: "no",
    });
    const loaded2 = await loadActionToken(db, minted.token);
    expect(loaded2?.redeemedValue).toBe("no");
  });
});
