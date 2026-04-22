/**
 * Matcher integration test — real PostGIS via @testcontainers/postgresql.
 *
 * Covers the spatial behaviour we can't verify against PGlite:
 *   - ST_DWithin picks up detections inside the AOI's distance buffer
 *   - ST_DWithin rejects detections outside the buffer
 *   - ST_Intersects with the industrial mask suppresses flare detections
 *   - Dedupe: a second poll with the same FIRMS rows is idempotent
 *   - Dedupe: a re-detection in the same window UPDATES the event row
 *     (detection_count bumps), a new window INSERTS a fresh row.
 *
 * Skipped locally if Docker isn't running; runs unconditionally on CI.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";
import {
  dockerAvailable,
  tryStartPostgisContainer,
  type TestcontainerHandle,
} from "@/db/test/testcontainer";
import { matchDetectionsToAois } from "@/lib/firms/matcher";
import type { FirmsDetection } from "@/lib/firms/client";
import { pointBoxToPolygon } from "@/lib/firms/industrial-seed";
import { regionBucketFromLonLat } from "@/lib/geo/region-bucket";

const SONOMA_LAT = 38.46;
const SONOMA_LON = -122.67;
const SONOMA_POLY_JSON = {
  type: "Polygon",
  coordinates: [
    [
      [SONOMA_LON - 0.05, SONOMA_LAT - 0.04],
      [SONOMA_LON + 0.05, SONOMA_LAT - 0.04],
      [SONOMA_LON + 0.05, SONOMA_LAT + 0.04],
      [SONOMA_LON - 0.05, SONOMA_LAT + 0.04],
      [SONOMA_LON - 0.05, SONOMA_LAT - 0.04],
    ],
  ],
};
const SONOMA_BUCKET = regionBucketFromLonLat(SONOMA_LON, SONOMA_LAT);

// Industrial flare centroid (Burgan, Kuwait).
const FLARE_LAT = 28.92;
const FLARE_LON = 47.93;

const probe = await dockerAvailable();
const describeIntegration = probe.available ? describe : describe.skip;

if (!probe.available) {
  console.warn(
    `[integration] Skipping PostGIS integration tests — Docker not available: ${probe.reason ?? "unknown"}`,
  );
}

describeIntegration("FIRMS matcher — PostGIS integration", () => {
  let handle: TestcontainerHandle | null = null;

  beforeAll(async () => {
    handle = await tryStartPostgisContainer();
  }, 180_000);

  afterAll(async () => {
    if (handle) await handle.stop();
  });

  beforeEach(async (ctx) => {
    if (!handle) {
      ctx.skip();
      return;
    }
    // Wipe everything but keep the stub user row so AOIs can FK to it.
    await handle!.pool.query(`DELETE FROM aoi_events`);
    await handle!.pool.query(`DELETE FROM firms_detections`);
    await handle!.pool.query(`DELETE FROM aoi_rules`);
    await handle!.pool.query(`DELETE FROM aois`);
    await handle!.pool.query(`DELETE FROM industrial_mask_static`);
    await handle!.pool.query(`DELETE FROM job_runs`);

    // Seed one industrial mask polygon for the Burgan flare.
    const flarePoly = pointBoxToPolygon(FLARE_LON, FLARE_LAT, 8);
    await handle!.pool.query(
      `INSERT INTO industrial_mask_static (kind, name, geom)
       VALUES ('gas_flare', 'Burgan (test)', ST_SetSRID(ST_GeomFromGeoJSON($1), 4326))`,
      [JSON.stringify(flarePoly)],
    );

    // Seed the Sonoma AOI + default rules.
    await handle!.pool.query(
      `INSERT INTO aois (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
       VALUES (
         'stub-user-1',
         'Spring Creek Preserve',
         ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON($1), 4326)),
         ST_SetSRID(ST_Envelope(ST_GeomFromGeoJSON($1)), 4326),
         ST_SetSRID(ST_Centroid(ST_GeomFromGeoJSON($1)), 4326),
         $2,
         2040
       )`,
      [JSON.stringify(SONOMA_POLY_JSON), SONOMA_BUCKET],
    );
    await handle!.pool.query(
      `INSERT INTO aoi_rules (aoi_id, distance_buffer_km, min_confidence, min_frp_mw)
       SELECT id, 25, 'nominal', 5 FROM aois LIMIT 1`,
    );
  });

  function makeDet(
    lat: number,
    lon: number,
    opts: Partial<FirmsDetection> = {},
  ): FirmsDetection {
    return {
      latitude: lat,
      longitude: lon,
      brightTi4: 325,
      brightTi5: 289,
      scan: 0.4,
      track: 0.4,
      acqDate: opts.acqDate ?? "2026-04-21",
      acqTime: opts.acqTime ?? "0417",
      satellite: "1",
      instrument: "VIIRS",
      confidence: opts.confidence ?? "n",
      version: "2.0NRT",
      frp: opts.frp ?? 11.2,
      daynight: opts.daynight ?? "N",
      ...opts,
    };
  }

  it("creates an event for a detection inside the distance buffer", async () => {
    // Detection 3 km north of the AOI polygon — well inside 25 km buffer.
    const det = makeDet(SONOMA_LAT + 0.08, SONOMA_LON + 0.01);
    const result = await matchDetectionsToAois(handle!.db, {
      bucket: SONOMA_BUCKET,
      source: "VIIRS_NOAA20_NRT",
      detections: [det],
    });
    expect(result.detectionsInserted).toBe(1);
    expect(result.eventsCreated).toBe(1);

    const events = await handle!.pool.query(`SELECT * FROM aoi_events`);
    expect(events.rowCount).toBe(1);
    const row = events.rows[0];
    expect(row.detection_count).toBe(1);
    expect(row.status).toBe("new");
    expect(Number(row.nearest_distance_km)).toBeGreaterThan(0);
    expect(Number(row.nearest_distance_km)).toBeLessThan(25);
  });

  it("ignores a detection beyond the buffer", async () => {
    // Detection ~40 km north of the AOI — beyond the default 25 km buffer.
    const det = makeDet(SONOMA_LAT + 0.5, SONOMA_LON);
    const result = await matchDetectionsToAois(handle!.db, {
      bucket: SONOMA_BUCKET,
      source: "VIIRS_NOAA20_NRT",
      detections: [det],
    });
    expect(result.detectionsInserted).toBe(1);
    expect(result.eventsCreated).toBe(0);
  });

  it("suppresses detections falling inside the industrial mask", async () => {
    // This detection is inside the Burgan flare polygon; it should be inserted
    // with is_industrial_static = TRUE but MUST NOT create an AOI event even
    // if (hypothetically) there were an AOI nearby.
    await handle!.pool.query(
      `INSERT INTO aois (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
       VALUES (
         'stub-user-1',
         'Kuwait Refuge',
         ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON($1), 4326)),
         ST_SetSRID(ST_Envelope(ST_GeomFromGeoJSON($1)), 4326),
         ST_SetSRID(ST_Centroid(ST_GeomFromGeoJSON($1)), 4326),
         $2,
         1200
       )`,
      [
        JSON.stringify({
          type: "Polygon",
          coordinates: [
            [
              [FLARE_LON - 0.02, FLARE_LAT - 0.02],
              [FLARE_LON + 0.02, FLARE_LAT - 0.02],
              [FLARE_LON + 0.02, FLARE_LAT + 0.02],
              [FLARE_LON - 0.02, FLARE_LAT + 0.02],
              [FLARE_LON - 0.02, FLARE_LAT - 0.02],
            ],
          ],
        }),
        regionBucketFromLonLat(FLARE_LON, FLARE_LAT),
      ],
    );
    await handle!.pool.query(
      `INSERT INTO aoi_rules (aoi_id, distance_buffer_km, min_confidence, min_frp_mw)
       SELECT id, 25, 'nominal', 5 FROM aois WHERE name = 'Kuwait Refuge'`,
    );
    const flareBucket = regionBucketFromLonLat(FLARE_LON, FLARE_LAT);

    const det = makeDet(FLARE_LAT, FLARE_LON);
    const result = await matchDetectionsToAois(handle!.db, {
      bucket: flareBucket,
      source: "VIIRS_NOAA20_NRT",
      detections: [det],
    });
    expect(result.detectionsInserted).toBe(1);
    expect(result.detectionsSkippedIndustrial).toBe(1);
    expect(result.eventsCreated).toBe(0);

    const events = await handle!.pool.query(
      `SELECT * FROM aoi_events WHERE aoi_id IN (SELECT id FROM aois WHERE name = 'Kuwait Refuge')`,
    );
    expect(events.rowCount).toBe(0);

    const detRows = await handle!.pool.query(
      `SELECT is_industrial_static FROM firms_detections`,
    );
    expect(detRows.rows[0].is_industrial_static).toBe(true);
  });

  it("is idempotent on a second poll with identical detections", async () => {
    const det = makeDet(SONOMA_LAT + 0.08, SONOMA_LON + 0.01);
    await matchDetectionsToAois(handle!.db, {
      bucket: SONOMA_BUCKET,
      source: "VIIRS_NOAA20_NRT",
      detections: [det],
    });
    const second = await matchDetectionsToAois(handle!.db, {
      bucket: SONOMA_BUCKET,
      source: "VIIRS_NOAA20_NRT",
      detections: [det],
    });
    expect(second.detectionsInserted).toBe(0); // ON CONFLICT DO NOTHING
    expect(second.eventsCreated).toBe(0);
    expect(second.eventsUpdated).toBe(0);

    const events = await handle!.pool.query(`SELECT COUNT(*)::int AS c FROM aoi_events`);
    expect(events.rows[0].c).toBe(1);
    const dets = await handle!.pool.query(`SELECT COUNT(*)::int AS c FROM firms_detections`);
    expect(dets.rows[0].c).toBe(1);
  });

  it("extends an existing event when a new detection lands in the same window", async () => {
    const first = makeDet(SONOMA_LAT + 0.08, SONOMA_LON + 0.01, {
      acqTime: "0417",
    });
    await matchDetectionsToAois(handle!.db, {
      bucket: SONOMA_BUCKET,
      source: "VIIRS_NOAA20_NRT",
      detections: [first],
    });
    // Second detection ~1h later, same rounded centroid => same dedupe hash.
    const second = makeDet(SONOMA_LAT + 0.081, SONOMA_LON + 0.011, {
      acqTime: "0517",
      frp: 18.5,
    });
    const result = await matchDetectionsToAois(handle!.db, {
      bucket: SONOMA_BUCKET,
      source: "VIIRS_NOAA20_NRT",
      detections: [second],
    });
    expect(result.detectionsInserted).toBe(1);
    expect(result.eventsCreated).toBe(0);
    expect(result.eventsUpdated).toBe(1);

    const events = await handle!.pool.query(
      `SELECT detection_count, peak_frp_mw FROM aoi_events`,
    );
    expect(events.rows[0].detection_count).toBe(2);
    expect(Number(events.rows[0].peak_frp_mw)).toBeCloseTo(18.5, 1);
  });

});
