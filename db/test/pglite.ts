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
import { readdirSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { makePgliteDb, type AppDb } from "@/lib/db/client";

let cachedDDL: string | null = null;

async function loadDDL(): Promise<string> {
  if (cachedDDL) return cachedDDL;
  const dir = join(process.cwd(), "db", "migrations");
  // Enumerate PGlite-compatible migration variants dynamically. Filename
  // pattern `NNNN_*.test.sql` is zero-padded so lexicographic sort == numeric.
  const files = readdirSync(dir)
    .filter((f) => f.endsWith(".test.sql"))
    .sort();
  const parts = await Promise.all(
    files.map((f) => readFile(join(dir, f), "utf8")),
  );
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
