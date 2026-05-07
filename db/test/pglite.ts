/**
 * PGlite test fixture — spins up a fresh in-memory Postgres per test, applies
 * the `*.test.sql` migrations (the PostGIS-free DDL), and returns a Drizzle
 * client configured to take the `usePostGIS = false` code path.
 *
 * Why two SQL files per migration: PGlite has no PostGIS extension, so the
 * production DDL cannot run as-is. The test variant stores GeoJSON in TEXT
 * columns; the repository code branches on `db.usePostGIS` to keep the
 * contract identical. Spatial tests (ST_Intersects, ST_DWithin) use the
 * @testcontainers/postgresql harness instead — see `db/test/testcontainer.ts`.
 */
import { PGlite } from "@electric-sql/pglite";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { makePgliteDb, type AppDb } from "@/lib/db/client";

/** Order matters — Stage N depends on Stage N-1. */
const MIGRATIONS = [
  "0000_init.test.sql",
  "0001_stage2.test.sql",
  "0002_stage3.test.sql",
  "0003_stage4.test.sql",
  "0004_stage5.test.sql",
  "0005_stage7.test.sql",
  "0006_stage8.test.sql",
] as const;

let cachedDDL: string | null = null;

async function loadDDL(): Promise<string> {
  if (cachedDDL) return cachedDDL;
  const parts: string[] = [];
  for (const file of MIGRATIONS) {
    const path = join(process.cwd(), "db", "migrations", file);
    parts.push(await readFile(path, "utf8"));
  }
  cachedDDL = parts.join("\n");
  return cachedDDL;
}

export async function makeFreshTestDb(): Promise<{ db: AppDb; pglite: PGlite }> {
  const pglite = new PGlite();
  await pglite.waitReady;
  const ddl = await loadDDL();
  await pglite.exec(ddl);
  const db = makePgliteDb(pglite);
  return { db, pglite };
}
