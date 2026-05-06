/**
 * /api/aoi/poll auth + build-without-blocking tests.
 *
 * These exercise the route handler's pre-DB branches (env var gates, bearer
 * auth, body validation) — they don't need PostGIS and run against PGlite
 * for the happy-path empty-bucket case.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { makeFreshTestDb } from "@/db/test/pglite";
import { _setTestDb, type AppDb } from "@/lib/db/client";
import type { PGlite } from "@electric-sql/pglite";
import { POST as pollPost } from "@/app/api/aoi/poll/route";

function req(headers: Record<string, string> = {}, body?: unknown): Request {
  return new Request("http://localhost/api/aoi/poll", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...headers,
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

const SAVED_ENV = { ...process.env };

describe("/api/aoi/poll — auth and config gates", () => {
  let pglite: PGlite;
  let db: AppDb;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
    _setTestDb(db);
    process.env.CRON_SECRET = "test-cron-secret";
    process.env.FIRMS_MAP_KEY = "test-firms-key";
  });

  afterEach(async () => {
    _setTestDb(null);
    await pglite.close();
    process.env = { ...SAVED_ENV };
  });

  it("returns 503 when CRON_SECRET is unset", async () => {
    delete process.env.CRON_SECRET;
    const res = await pollPost(
      req({ authorization: "Bearer anything" }) as Parameters<typeof pollPost>[0],
    );
    expect(res.status).toBe(503);
    const body = (await res.json()) as { error: { code: string } };
    expect(body.error.code).toBe("service_unavailable");
  });

  it("returns 503 when FIRMS_MAP_KEY is unset (even with valid auth)", async () => {
    delete process.env.FIRMS_MAP_KEY;
    const res = await pollPost(
      req({ authorization: "Bearer test-cron-secret" }) as Parameters<typeof pollPost>[0],
    );
    expect(res.status).toBe(503);
  });

  it("rejects missing bearer with 400", async () => {
    const res = await pollPost(req() as Parameters<typeof pollPost>[0]);
    expect(res.status).toBe(400);
  });

  it("rejects a wrong bearer with 401", async () => {
    const res = await pollPost(
      req({ authorization: "Bearer wrong-secret" }) as Parameters<typeof pollPost>[0],
    );
    expect(res.status).toBe(401);
  });

  it("rejects a malformed body with 400", async () => {
    const res = await pollPost(
      req(
        { authorization: "Bearer test-cron-secret" },
        { bucket: "not-a-bucket-key" },
      ) as Parameters<typeof pollPost>[0],
    );
    expect(res.status).toBe(400);
  });

  it("returns 200 and an empty run list when no active AOIs exist", async () => {
    const res = await pollPost(
      req({ authorization: "Bearer test-cron-secret" }, {}) as Parameters<typeof pollPost>[0],
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      runs: unknown[];
      bucketCount: number;
    };
    expect(body.runs).toEqual([]);
    expect(body.bucketCount).toBe(0);
  });
});
