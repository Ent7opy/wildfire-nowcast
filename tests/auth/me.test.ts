/**
 * Stage 5 — `/api/me` route shape coverage.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import { _setTestDb, type AppDb } from "@/lib/db/client";
import { _setTestAuth } from "@/lib/auth/context";
import type { PGlite } from "@electric-sql/pglite";
import { GET as meGet } from "@/app/api/me/route";

describe("/api/me", () => {
  let db: AppDb;
  let pglite: PGlite;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
    _setTestDb(db);
  });
  afterEach(async () => {
    _setTestDb(null);
    _setTestAuth(null);
    await pglite.close();
  });

  it("returns 401 when not signed in", async () => {
    _setTestAuth(() => ({ ok: false, code: "unauthenticated" }));
    const res = await meGet();
    expect(res.status).toBe(401);
  });

  it("returns id, email, hasByoKey for the authed user", async () => {
    _setTestAuth(() => ({ ok: true, userId: "user_2abcMe" }));
    // Pre-populate so the JIT path lands the email upgrade does not race the assert.
    await db.execute(sql`
      INSERT INTO users (id, email) VALUES ('user_2abcMe', 'me@example.org')
    `);
    const res = await meGet();
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      id: string;
      email: string;
      hasByoKey: boolean;
    };
    expect(body.id).toBe("user_2abcMe");
    expect(body.email).toBe("me@example.org");
    expect(body.hasByoKey).toBe(false);
  });
});
