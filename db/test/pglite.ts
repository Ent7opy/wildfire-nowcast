/**
 * PGlite test fixture — spins up a fresh in-memory Postgres per test, applies
 * `db/migrations/0000_init.test.sql` (the PostGIS-free DDL), and returns a
 * Drizzle client configured to take the `usePostGIS = false` code path.
 *
 * Why two SQL files: PGlite has no PostGIS extension, so the production DDL
 * cannot run as-is. The test variant stores GeoJSON in TEXT columns; the
 * repository code branches on `db.usePostGIS` to keep the contract identical.
 */
import { PGlite } from "@electric-sql/pglite";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { makePgliteDb, type AppDb } from "@/lib/db/client";

const MIGRATION_PATH = join(
  process.cwd(),
  "db",
  "migrations",
  "0000_init.test.sql",
);

let cachedDDL: string | null = null;

async function loadDDL(): Promise<string> {
  if (cachedDDL) return cachedDDL;
  cachedDDL = await readFile(MIGRATION_PATH, "utf8");
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
