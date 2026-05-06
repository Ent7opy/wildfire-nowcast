/**
 * Stage 7 — listMatchedDetectionsForAoi reads the right shape, honors the
 * 90-day window, and ownership-checks against userId.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import { listMatchedDetectionsForAoi } from "@/lib/db/aoi-repository";
import { seedAoi, seedUser, SOFIA } from "./_helpers";
import type { PGlite } from "@electric-sql/pglite";

async function insertEvent(
  db: AppDb,
  args: { aoiId: string; firstSeen: string; lastSeen: string },
): Promise<void> {
  await db.execute(sql`
    INSERT INTO "aoi_events" (aoi_id, first_seen_at, last_seen_at, nearest_distance_km,
      detection_count, dedupe_hash, status)
    VALUES (${args.aoiId}, ${args.firstSeen}, ${args.lastSeen}, 1, 1,
      ${"h-" + Math.random()}, 'new')
  `);
}

async function insertDetection(
  db: AppDb,
  args: {
    detectedAt: string;
    lat: number;
    lon: number;
    bucket: string;
    industrial?: boolean;
  },
): Promise<void> {
  const point = JSON.stringify({ type: "Point", coordinates: [args.lon, args.lat] });
  await db.execute(sql`
    INSERT INTO "firms_detections" (
      "source", "detected_at", "geom", "lat", "lon",
      "acq_date", "acq_time", "bucket", "is_industrial_static"
    ) VALUES (
      'VIIRS_NOAA20_NRT', ${args.detectedAt}, ${point},
      ${args.lat}, ${args.lon},
      ${args.detectedAt.slice(0, 10)}, '0000', ${args.bucket},
      ${args.industrial ?? false}
    )
  `);
}

describe("listMatchedDetectionsForAoi", () => {
  let db: AppDb;
  let pglite: PGlite;
  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
  });
  afterEach(async () => {
    await pglite.close();
  });

  it("returns detections that fall inside an event window", async () => {
    await seedUser(db, "user_alice");
    const aoiId = await seedAoi(db, "user_alice", "Sofia", SOFIA);
    const now = new Date("2026-04-21T00:00:00Z");
    const eventStart = new Date(now.getTime() - 5 * 86400_000).toISOString();
    const eventEnd = new Date(now.getTime() - 4 * 86400_000).toISOString();
    await insertEvent(db, { aoiId, firstSeen: eventStart, lastSeen: eventEnd });
    // Read the AOI's region bucket to insert detections in that bucket
    const r = (await db.execute(
      sql`SELECT "region_bucket" FROM "aois" WHERE "id" = ${aoiId}`,
    )) as unknown as { rows?: Array<{ region_bucket: string }> };
    const bucket = ((r.rows ?? r) as Array<{ region_bucket: string }>)[0].region_bucket;
    // Inside window
    await insertDetection(db, {
      detectedAt: new Date(new Date(eventStart).getTime() + 3600_000).toISOString(),
      lat: 42.7, lon: 23.35, bucket,
    });
    // Outside window (older than event start)
    await insertDetection(db, {
      detectedAt: new Date(new Date(eventStart).getTime() - 86400_000).toISOString(),
      lat: 42.71, lon: 23.36, bucket,
    });
    const out = await listMatchedDetectionsForAoi(db, {
      userId: "user_alice", aoiId, sinceDays: 90, now,
    });
    expect(out).toHaveLength(1);
    expect(out[0].satellite).toBe("VIIRS_NOAA20_NRT");
  });

  it("honors the 90-day cap even when the event predates it", async () => {
    await seedUser(db, "user_bob");
    const aoiId = await seedAoi(db, "user_bob", "Sofia", SOFIA);
    const now = new Date("2026-04-21T00:00:00Z");
    const old = new Date(now.getTime() - 200 * 86400_000).toISOString();
    await insertEvent(db, { aoiId, firstSeen: old, lastSeen: old });
    const r = (await db.execute(
      sql`SELECT "region_bucket" FROM "aois" WHERE "id" = ${aoiId}`,
    )) as unknown as { rows?: Array<{ region_bucket: string }> };
    const bucket = ((r.rows ?? r) as Array<{ region_bucket: string }>)[0].region_bucket;
    await insertDetection(db, { detectedAt: old, lat: 42.7, lon: 23.35, bucket });
    const out = await listMatchedDetectionsForAoi(db, {
      userId: "user_bob", aoiId, sinceDays: 90, now,
    });
    expect(out).toHaveLength(0);
  });

  it("returns empty when AOI is not owned by the user", async () => {
    await seedUser(db, "user_owner");
    await seedUser(db, "user_intruder");
    const aoiId = await seedAoi(db, "user_owner", "Sofia", SOFIA);
    const out = await listMatchedDetectionsForAoi(db, {
      userId: "user_intruder", aoiId,
    });
    expect(out).toEqual([]);
  });
});
