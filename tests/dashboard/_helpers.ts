/**
 * Test helpers — seed AOIs + briefs into a PGlite-backed AppDb.
 */
import { sql } from "drizzle-orm";
import type { AppDb } from "@/lib/db/client";
import { createAoi } from "@/lib/db/aoi-repository";
import type { PolygonalGeom } from "@/lib/validators/geojson";

export const SONOMA: PolygonalGeom = {
  type: "Polygon",
  coordinates: [
    [
      [-122.72, 38.42],
      [-122.62, 38.42],
      [-122.62, 38.5],
      [-122.72, 38.5],
      [-122.72, 38.42],
    ],
  ],
};

export const SOFIA: PolygonalGeom = {
  type: "Polygon",
  coordinates: [
    [
      [23.3, 42.65],
      [23.4, 42.65],
      [23.4, 42.75],
      [23.3, 42.75],
      [23.3, 42.65],
    ],
  ],
};

export async function seedUser(db: AppDb, userId: string, email = `${userId}@x`): Promise<void> {
  await db.execute(sql`
    INSERT INTO "users" ("id", "email") VALUES (${userId}, ${email})
    ON CONFLICT ("id") DO NOTHING
  `);
}

export async function seedAoi(
  db: AppDb,
  userId: string,
  name: string,
  geom: PolygonalGeom = SONOMA,
): Promise<string> {
  const out = await createAoi(db, { userId, name, geometry: geom });
  return out.aoi.id;
}

/**
 * Insert an event + brief and return the brief id. Includes a synthetic
 * payload with a `summary` field so CSV-export tests have something to read.
 */
export async function seedBrief(
  db: AppDb,
  args: {
    aoiId: string;
    createdAt?: Date;
    summary?: string;
    gateReason?: string;
  },
): Promise<string> {
  const createdAt = args.createdAt ?? new Date();
  const payload = {
    summary: args.summary ?? "Synthetic brief summary",
  };
  const eventResult = (await db.execute(sql`
    INSERT INTO "aoi_events" (
      "aoi_id", "first_seen_at", "last_seen_at", "nearest_distance_km",
      "detection_count", "dedupe_hash", "status"
    ) VALUES (
      ${args.aoiId}, ${createdAt.toISOString()}, ${createdAt.toISOString()},
      1.5, 1, ${"hash-" + Math.random()}, 'new'
    )
    RETURNING "id"
  `)) as unknown as { rows?: Array<{ id: string }> };
  const eventRows = (eventResult.rows ?? (eventResult as unknown as Array<{ id: string }>));
  const eventId = eventRows[0].id;

  const briefResult = (await db.execute(sql`
    INSERT INTO "aoi_briefs" (
      "aoi_id", "event_id", "model", "gate_reason", "payload",
      "rendered_markdown", "created_at"
    ) VALUES (
      ${args.aoiId}, ${eventId},
      'google/gemini-2.5-flash-lite',
      ${args.gateReason ?? "polygon_intersect"},
      ${JSON.stringify(payload)}::jsonb,
      ${"# Test brief\n\n" + (args.summary ?? "Synthetic")},
      ${createdAt.toISOString()}
    )
    RETURNING "id"
  `)) as unknown as { rows?: Array<{ id: string }> };
  const briefRows = (briefResult.rows ?? (briefResult as unknown as Array<{ id: string }>));
  return briefRows[0].id;
}
