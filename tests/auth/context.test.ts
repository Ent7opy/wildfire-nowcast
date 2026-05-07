/**
 * Stage 5 — `requireUserId` branch coverage.
 *
 * Uses `_setTestAuth` to bypass Clerk's runtime; covers the env-gate path and
 * the success/failure shapes the route handlers depend on.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { _setTestAuth, ensureUserExists, requireUserId } from "@/lib/auth/context";
import { makeFreshTestDb } from "@/db/test/pglite";

const SAVED_ENV = { ...process.env };

describe("requireUserId", () => {
  beforeEach(() => {
    _setTestAuth(null);
  });
  afterEach(() => {
    _setTestAuth(null);
    process.env = { ...SAVED_ENV };
  });

  it("returns config_missing when CLERK_SECRET_KEY is unset and no test auth", async () => {
    delete process.env.CLERK_SECRET_KEY;
    const r = await requireUserId();
    expect(r).toEqual({ ok: false, code: "config_missing" });
  });

  it("returns the test-injected userId regardless of env", async () => {
    delete process.env.CLERK_SECRET_KEY;
    _setTestAuth(() => ({ ok: true, userId: "user_2abcStubbed" }));
    const r = await requireUserId();
    expect(r).toEqual({ ok: true, userId: "user_2abcStubbed" });
  });

  it("returns unauthenticated when test auth says so", async () => {
    _setTestAuth(() => ({ ok: false, code: "unauthenticated" }));
    const r = await requireUserId();
    expect(r).toEqual({ ok: false, code: "unauthenticated" });
  });
});

describe("ensureUserExists JIT INSERT satisfies users NOT-NULL constraints", () => {
  it("populates every NOT-NULL column without a DB default", async () => {
    const { db, pglite } = await makeFreshTestDb();
    try {
      await ensureUserExists(db, "user_2abcSchemaDriven");
      const colsRes = (await db.execute(sql`
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'users' AND is_nullable = 'NO' AND column_default IS NULL
      `)) as unknown as { rows?: Array<{ column_name: string }> };
      const required = (colsRes.rows ?? (colsRes as unknown as Array<{ column_name: string }>))
        .map((r) => r.column_name);
      expect(required.length).toBeGreaterThan(0);
      const rowRes = (await db.execute(
        sql`SELECT * FROM users WHERE id = 'user_2abcSchemaDriven'`,
      )) as unknown as { rows?: Array<Record<string, unknown>> };
      const row = (rowRes.rows ?? (rowRes as unknown as Array<Record<string, unknown>>))[0];
      expect(row).toBeDefined();
      for (const col of required) {
        expect(row[col], `JIT INSERT did not populate NOT-NULL column "${col}"`).not.toBeNull();
      }
    } finally {
      await pglite.close();
    }
  });
});
