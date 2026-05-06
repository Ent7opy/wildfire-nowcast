/**
 * Seeds `industrial_mask_static` from `db/seeds/industrial-mask-stage2.json`.
 *
 * Idempotent via the (kind, name) unique index — re-running re-inserts only
 * polygons that were added or renamed since the last run.
 *
 * Usage:
 *   pnpm db:seed:industrial          # uses .env / DATABASE_URL
 */
import "dotenv/config";
import { Client } from "pg";
import {
  loadIndustrialMaskSeed,
  pointBoxToPolygon,
} from "@/lib/firms/industrial-seed";

async function main(): Promise<void> {
  const url = process.env.DATABASE_URL;
  if (!url) {
    console.error("DATABASE_URL is not set; nothing to do.");
    process.exit(1);
  }
  const seed = await loadIndustrialMaskSeed();
  const client = new Client({ connectionString: url });
  await client.connect();
  let inserted = 0;
  try {
    for (const row of seed.polygons) {
      const polygon = pointBoxToPolygon(row.lon, row.lat, row.radiusKm);
      const result = await client.query(
        `INSERT INTO industrial_mask_static (kind, name, geom, source_url)
         VALUES ($1, $2, ST_SetSRID(ST_GeomFromGeoJSON($3), 4326), $4)
         ON CONFLICT (kind, name) DO NOTHING`,
        [row.kind, row.name, JSON.stringify(polygon), null],
      );
      inserted += result.rowCount ?? 0;
    }
    console.log(
      `Industrial mask seed: ${inserted} new rows inserted (${seed.polygons.length} total in source).`,
    );
  } finally {
    await client.end();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
