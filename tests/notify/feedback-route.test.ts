/**
 * Stage 7 follow-up — verifies the feedback route validates `?v=` BEFORE
 * consuming the action token. Reviewer feedback on PR #406: a click without
 * `?v=` should not mark the token redeemed; the user's next click with a
 * real value must succeed via the first-redemption path.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import {
  loadActionToken,
  mintActionToken,
} from "@/lib/notify/action-tokens";
import type { PGlite } from "@electric-sql/pglite";

let currentDb: AppDb | null = null;

vi.mock("@/lib/db/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/db/client")>();
  return {
    ...actual,
    tryGetDb: () => currentDb,
  };
});

async function seedFeedbackToken(
  db: AppDb,
): Promise<{ token: string; aoiId: string; briefId: string }> {
  const userId = `u-${Math.random().toString(36).slice(2, 8)}`;
  await db.execute(
    sql`INSERT INTO "users" (id, email) VALUES (${userId}, ${userId + "@x"})`,
  );
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
  const aoiId = ((aoi.rows ?? aoi) as Array<{ id: string }>)[0].id;
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
  const briefId = ((brief.rows ?? brief) as Array<{ id: string }>)[0].id;
  const minted = await mintActionToken(db, {
    aoiId,
    briefId,
    action: "feedback",
    channel: "email",
    target: "alice@example.org",
  });
  return { token: minted.token, aoiId, briefId };
}

describe("feedback route — validates ?v= before consuming token", () => {
  let db: AppDb;
  let pglite: PGlite;
  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
    currentDb = db;
  });
  afterEach(async () => {
    currentDb = null;
    await pglite.close();
  });

  it("missing ?v= returns 400 and does NOT mark the token consumed", async () => {
    const { token } = await seedFeedbackToken(db);
    const { GET } = await import("@/app/api/notify/feedback/[token]/route");
    const req = new Request(`http://localhost/api/notify/feedback/${token}`);
    const res = await GET(req as unknown as import("next/server").NextRequest, {
      params: Promise.resolve({ token }),
    });
    expect(res.status).toBe(400);
    const loaded = await loadActionToken(db, token);
    expect(loaded).not.toBeNull();
    expect(loaded?.redeemedAt).toBeNull();
    expect(loaded?.redeemedValue).toBeNull();
  });

  it("after a missing-value click, a follow-up ?v=yes succeeds via first-redemption path", async () => {
    const { token } = await seedFeedbackToken(db);
    const { GET } = await import("@/app/api/notify/feedback/[token]/route");

    const res1 = await GET(
      new Request(
        `http://localhost/api/notify/feedback/${token}`,
      ) as unknown as import("next/server").NextRequest,
      { params: Promise.resolve({ token }) },
    );
    expect(res1.status).toBe(400);

    const res2 = await GET(
      new Request(
        `http://localhost/api/notify/feedback/${token}?v=yes`,
      ) as unknown as import("next/server").NextRequest,
      { params: Promise.resolve({ token }) },
    );
    expect(res2.status).toBe(200);
    const loaded = await loadActionToken(db, token);
    expect(loaded?.redeemedValue).toBe("yes");
    expect(loaded?.redeemedAt).not.toBeNull();
  });

  it("invalid ?v=maybe returns 400 and does NOT mark the token consumed", async () => {
    const { token } = await seedFeedbackToken(db);
    const { GET } = await import("@/app/api/notify/feedback/[token]/route");
    const res = await GET(
      new Request(
        `http://localhost/api/notify/feedback/${token}?v=maybe`,
      ) as unknown as import("next/server").NextRequest,
      { params: Promise.resolve({ token }) },
    );
    expect(res.status).toBe(400);
    const loaded = await loadActionToken(db, token);
    expect(loaded?.redeemedAt).toBeNull();
  });
});
