/**
 * PostGIS testcontainer harness for spatial integration tests.
 *
 * Spins up `postgis/postgis:16-3.5` via @testcontainers/postgresql, applies
 * both migrations (Stage 1 + Stage 2, production variants), and returns a
 * Drizzle client tagged as `usePostGIS = true` so the repository takes the
 * production SQL path.
 *
 * Gated behind a Docker-availability probe: if Docker isn't reachable (most
 * common on a fresh laptop before Vanyo opens Docker Desktop), we return
 * `{ available: false }` and tests call `it.skip` with a clear message.
 *
 * CI (GitHub Actions Ubuntu runner) has Docker available by default, so the
 * integration tests run as part of `pnpm test` there. Local `pnpm test`
 * without Docker still passes — spatial tests skip with a visible notice.
 */
import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { promisify } from "node:util";
import { Pool } from "pg";
import { drizzle } from "drizzle-orm/node-postgres";
import * as schema from "@/db/schema";
import type { AppDb } from "@/lib/db/client";

const execFileP = promisify(execFile);

let dockerProbeCache: { available: boolean; reason?: string } | null = null;

/**
 * Fast, resilient Docker-availability probe. Runs `docker info` with a short
 * timeout; caches the result for the process lifetime.
 *
 * Returning `available: true` here means the daemon is reachable; pulling the
 * postgis image may still fail (e.g. arch mismatch, registry outage). Tests
 * call `tryStartPostgisContainer` which catches container-start failures and
 * lets the test suite skip gracefully.
 */
export async function dockerAvailable(): Promise<{ available: boolean; reason?: string }> {
  if (dockerProbeCache) return dockerProbeCache;
  try {
    await execFileP("docker", ["info"], { timeout: 3_000 });
    dockerProbeCache = { available: true };
  } catch (err) {
    dockerProbeCache = {
      available: false,
      reason: err instanceof Error ? err.message.split("\n")[0] : String(err),
    };
  }
  return dockerProbeCache;
}

/**
 * Try to start a PostGIS container, returning null on any failure. Used by
 * integration tests that want to skip cleanly on machines where the image
 * isn't available (e.g. arm64 dev boxes without a multi-arch postgis).
 */
export async function tryStartPostgisContainer(): Promise<TestcontainerHandle | null> {
  try {
    return await startPostgisContainer();
  } catch (err) {
    console.warn(
      `[integration] PostGIS container failed to start; skipping: ${
        err instanceof Error ? err.message.split("\n")[0] : String(err)
      }`,
    );
    return null;
  }
}

export type TestcontainerHandle = {
  db: AppDb;
  pool: Pool;
  stop: () => Promise<void>;
};

async function loadMigrations(): Promise<string> {
  const dir = join(process.cwd(), "db", "migrations");
  const stage1 = await readFile(join(dir, "0000_init.sql"), "utf8");
  const stage2 = await readFile(join(dir, "0001_stage2.sql"), "utf8");
  const stage3 = await readFile(join(dir, "0002_stage3.sql"), "utf8");
  return [stage1, stage2, stage3].join("\n");
}

/**
 * Start a fresh PostGIS container, apply migrations, return a Drizzle handle.
 * Safe to call only after `dockerAvailable()` returns true.
 *
 * Image: `imresamu/postgis:16-3.5-alpine` — a multi-arch fork of the upstream
 * postgis image. Upstream `postgis/postgis:16-3.5` is amd64-only as of writing
 * (apple silicon dev boxes need to fall back to qemu, which is slow and
 * occasionally broken). Override via env `WFN_POSTGIS_IMAGE`.
 */
export async function startPostgisContainer(): Promise<TestcontainerHandle> {
  // Lazy-import so the test file can import this module without crashing when
  // Docker is absent (the import of @testcontainers/postgresql succeeds; it's
  // the `.start()` call that would fail).
  const mod = await import("@testcontainers/postgresql");
  const image =
    process.env.WFN_POSTGIS_IMAGE ?? "imresamu/postgis:16-3.5-alpine";
  const container = await new mod.PostgreSqlContainer(image)
    .withDatabase("wildfire_test")
    .withUsername("wildfire")
    .withPassword("wildfire")
    .start();

  const url = container.getConnectionUri();
  const pool = new Pool({ connectionString: url, max: 4 });
  const migrations = await loadMigrations();
  await pool.query(migrations);

  const inner = drizzle(pool, { schema });
  const db = inner as unknown as AppDb;
  (db as { __backend: string }).__backend = "neon";
  (db as { usePostGIS: boolean }).usePostGIS = true;

  return {
    db,
    pool,
    stop: async () => {
      await pool.end();
      await container.stop();
    },
  };
}
