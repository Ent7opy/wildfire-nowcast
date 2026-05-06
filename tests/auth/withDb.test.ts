/**
 * Stage 5 — `withDb` branch coverage on PGlite.
 *
 * Exercises the auth + JIT-provisioning composition through the actual route
 * handlers rather than calling `withDb` directly.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import { _setTestDb, type AppDb } from "@/lib/db/client";
import { _setTestAuth } from "@/lib/auth/context";
import type { PGlite } from "@electric-sql/pglite";
import { GET as aoiList } from "@/app/api/aoi/route";

const SAVED_ENV = { ...process.env };

describe("withDb auth gate", () => {
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
    process.env = { ...SAVED_ENV };
  });

  it("returns 401 when auth says unauthenticated", async () => {
    _setTestAuth(() => ({ ok: false, code: "unauthenticated" }));
    const res = await aoiList();
    expect(res.status).toBe(401);
    const body = (await res.json()) as { error: { code: string } };
    expect(body.error.code).toBe("unauthenticated");
  });

  it("returns 503 when auth says config_missing", async () => {
    _setTestAuth(() => ({ ok: false, code: "config_missing" }));
    const res = await aoiList();
    expect(res.status).toBe(503);
    const body = (await res.json()) as { error: { code: string } };
    expect(body.error.code).toBe("service_unavailable");
  });

  it("JIT-provisions a users row on the first authed request", async () => {
    _setTestAuth(() => ({ ok: true, userId: "user_2abcJustInTime" }));
    const res = await aoiList();
    expect(res.status).toBe(200);
    const rows = (await db.execute(sql`SELECT id FROM users WHERE id = 'user_2abcJustInTime'`)) as unknown as {
      rows?: Array<{ id: string }>;
    };
    const r = (rows.rows ?? (rows as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
    expect(r).toHaveLength(1);
  });

  it("does not duplicate the users row on the second authed request", async () => {
    _setTestAuth(() => ({ ok: true, userId: "user_2abcStable" }));
    await aoiList();
    await aoiList();
    const rows = (await db.execute(sql`SELECT count(*)::int AS c FROM users WHERE id = 'user_2abcStable'`)) as unknown as {
      rows?: Array<{ c: number }>;
    };
    const r = (rows.rows ?? (rows as unknown as Array<{ c: number }>)) as Array<{ c: number }>;
    expect(r[0].c).toBe(1);
  });
});
