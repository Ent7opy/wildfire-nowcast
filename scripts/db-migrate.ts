/**
 * One-shot migrator for the live Neon database.
 *
 * Runs `db/migrations/0000_init.sql` against `DATABASE_URL`. Idempotent:
 * every CREATE statement uses IF NOT EXISTS, and the seed `INSERT` is
 * `ON CONFLICT DO NOTHING`. Safe to re-run.
 *
 * Usage:
 *   pnpm db:migrate                # uses .env / DATABASE_URL
 *
 * In Stage 5+ when more migrations land, replace this with `drizzle-kit
 * migrate` against `db/migrations/`. Stage 1 keeps the runner trivial.
 */
import "dotenv/config";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { Client } from "pg";

async function main(): Promise<void> {
  const url = process.env.DATABASE_URL;
  if (!url) {
    console.error("DATABASE_URL is not set; nothing to do.");
    process.exit(1);
  }
  const sqlPath = join(process.cwd(), "db", "migrations", "0000_init.sql");
  const sql = await readFile(sqlPath, "utf8");
  const client = new Client({ connectionString: url });
  await client.connect();
  try {
    await client.query(sql);
    console.log("Applied 0000_init.sql");
  } finally {
    await client.end();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
