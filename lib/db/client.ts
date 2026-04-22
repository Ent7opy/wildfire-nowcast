/**
 * Drizzle client. Two implementations behind one interface:
 *
 *   1. Production / preview: node-postgres `Pool` against Neon's pooled URL.
 *      `usePostGIS = true` — repository uses ST_* SQL functions.
 *   2. Tests: PGlite in-memory database. `usePostGIS = false` — repository
 *      reads/writes GeoJSON as TEXT and uses application-side geometry.
 *
 * The Stage 1 brief requires that the app *builds* without DATABASE_URL set
 * (Vanyo has not yet provisioned Neon). We achieve this by:
 *   - never connecting at import time
 *   - returning null from `tryGetDb()` when DATABASE_URL is absent
 *   - route handlers map null → 503 service_unavailable
 */
import { drizzle as drizzleNodePg, type NodePgDatabase } from "drizzle-orm/node-postgres";
import { drizzle as drizzlePglite, type PgliteDatabase } from "drizzle-orm/pglite";
import { Pool } from "pg";
import * as schema from "@/db/schema";

export type AppDb =
  | (NodePgDatabase<typeof schema> & { __backend: "neon"; usePostGIS: true })
  | (PgliteDatabase<typeof schema> & { __backend: "pglite"; usePostGIS: false });

let cached: AppDb | null = null;

export function tryGetDb(): AppDb | null {
  if (cached) return cached;
  const url = process.env.DATABASE_URL;
  if (!url) return null;
  const pool = new Pool({ connectionString: url, max: 5 });
  const inner = drizzleNodePg(pool, { schema });
  const db = inner as unknown as NodePgDatabase<typeof schema> & {
    __backend: "neon";
    usePostGIS: true;
  };
  db.__backend = "neon";
  db.usePostGIS = true;
  cached = db;
  return db;
}

/**
 * Test-only entry point. Wraps a PGlite instance in Drizzle and tags it as
 * non-PostGIS so the repository takes the GeoJSON-as-TEXT code path.
 *
 * Imported only by the test harness; production code should never reach this.
 */
export function makePgliteDb(
  pglite: ConstructorParameters<typeof import("@electric-sql/pglite").PGlite>[0] extends never
    ? never
    : import("@electric-sql/pglite").PGlite,
): AppDb {
  const inner = drizzlePglite(pglite, { schema });
  const db = inner as unknown as PgliteDatabase<typeof schema> & {
    __backend: "pglite";
    usePostGIS: false;
  };
  db.__backend = "pglite";
  db.usePostGIS = false;
  return db;
}

/**
 * For tests only: wipe the cache so a fresh db can be installed.
 */
export function _setTestDb(db: AppDb | null): void {
  cached = db;
}
