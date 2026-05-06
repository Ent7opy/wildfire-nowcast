/**
 * One-shot migrator for the live Neon database.
 *
 * Applies every `db/migrations/<NNNN>_*.sql` file (skipping `*.test.sql`
 * variants which are PGlite-only) in lexical order. Idempotent: every CREATE
 * uses IF NOT EXISTS and seeds use `ON CONFLICT DO NOTHING`. Safe to re-run.
 *
 * Usage:
 *   pnpm db:migrate                # uses .env / DATABASE_URL
 *
 * In Stage 5+ when migration count grows, replace with `drizzle-kit migrate`.
 */
import "dotenv/config";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { Client } from "pg";

async function main(): Promise<void> {
  const url = process.env.DATABASE_URL;
  if (!url) {
    console.error("DATABASE_URL is not set; nothing to do.");
    process.exit(1);
  }
  const dir = join(process.cwd(), "db", "migrations");
  const files = (await readdir(dir))
    .filter((f) => f.endsWith(".sql") && !f.endsWith(".test.sql"))
    .sort();
  const client = new Client({ connectionString: url });
  await client.connect();
  try {
    for (const file of files) {
      const sql = await readFile(join(dir, file), "utf8");
      await client.query(sql);
      console.log(`Applied ${file}`);
    }
  } finally {
    await client.end();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
