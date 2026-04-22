/**
 * drizzle-kit config — used to generate / introspect SQL migrations against
 * the live Neon database. PGlite-backed unit tests do not use this; they
 * apply `db/migrations/0000_init.sql` directly.
 *
 * Note: scripts that read DATABASE_URL must explicitly load .env.local —
 * Next.js auto-loads it at runtime, but drizzle-kit (run via tsx/pnpm) does
 * not. The `dotenv/config` import below handles that.
 */
import "dotenv/config";
import { defineConfig } from "drizzle-kit";

export default defineConfig({
  schema: "./db/schema/index.ts",
  out: "./db/migrations",
  dialect: "postgresql",
  dbCredentials: {
    url: process.env.DATABASE_URL ?? "postgres://placeholder/placeholder",
  },
  // We keep generated SQL alongside hand-edited PostGIS / index DDL.
  // See db/migrations/0000_init.sql for the canonical Stage 1 migration.
  verbose: true,
  strict: true,
});
