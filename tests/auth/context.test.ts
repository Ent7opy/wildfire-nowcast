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
import type { AppDb } from "@/lib/db/client";
import type { PGlite } from "@electric-sql/pglite";

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

  it("test injection short-circuits the env check even when CLERK_SECRET_KEY is set", async () => {
    // Pins precedence: a stubbed resolver must win over the production path so
    // tests never accidentally hit Clerk just because an env leaked into CI.
    process.env.CLERK_SECRET_KEY = "sk_test_should_not_be_read";
    _setTestAuth(() => ({ ok: true, userId: "user_precedence" }));
    const r = await requireUserId();
    expect(r).toEqual({ ok: true, userId: "user_precedence" });
  });

  it("awaits an async resolver (Promise<AuthResult>)", async () => {
    // The TestAuthFn signature allows sync or async; production Clerk `auth()`
    // is async, so the async branch must work end-to-end.
    _setTestAuth(async () => {
      await Promise.resolve();
      return { ok: true, userId: "user_async" };
    });
    const r = await requireUserId();
    expect(r).toEqual({ ok: true, userId: "user_async" });
  });

  it("clearing the test resolver with null restores the env-gated path", async () => {
    _setTestAuth(() => ({ ok: true, userId: "user_will_be_cleared" }));
    _setTestAuth(null);
    delete process.env.CLERK_SECRET_KEY;
    const r = await requireUserId();
    expect(r).toEqual({ ok: false, code: "config_missing" });
  });
});

describe("ensureUserExists", () => {
  let db: AppDb;
  let pglite: PGlite;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
  });
  afterEach(async () => {
    await pglite.close();
  });

  it("inserts a row with the <userId>@pending.invalid placeholder email", async () => {
    // The placeholder shape is load-bearing: the dispatcher special-cases
    // `@pending.invalid` to skip with `no_recipient_pending`. Pin the contract
    // here so a refactor of the placeholder format trips this test before it
    // silently breaks notification gating.
    await ensureUserExists(db, "user_2abcJIT");
    const result = (await db.execute(
      sql`SELECT id, email FROM users WHERE id = 'user_2abcJIT'`,
    )) as unknown as { rows?: Array<{ id: string; email: string }> };
    const rows = (result.rows ??
      (result as unknown as Array<{ id: string; email: string }>)) as Array<{
      id: string;
      email: string;
    }>;
    expect(rows).toHaveLength(1);
    expect(rows[0].email).toBe("user_2abcJIT@pending.invalid");
  });

  it("is idempotent and does not overwrite an existing email on conflict", async () => {
    // Webhook-then-JIT ordering: if `user.created` lands first with a real
    // email, a follow-up authed request must not stomp it back to placeholder.
    await db.execute(
      sql`INSERT INTO users (id, email) VALUES ('user_existing', 'real@example.com')`,
    );
    await ensureUserExists(db, "user_existing");
    await ensureUserExists(db, "user_existing");
    const result = (await db.execute(
      sql`SELECT email, count(*)::int AS c FROM users WHERE id = 'user_existing' GROUP BY email`,
    )) as unknown as { rows?: Array<{ email: string; c: number }> };
    const rows = (result.rows ??
      (result as unknown as Array<{ email: string; c: number }>)) as Array<{
      email: string;
      c: number;
    }>;
    expect(rows).toHaveLength(1);
    expect(rows[0].email).toBe("real@example.com");
    expect(rows[0].c).toBe(1);
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
