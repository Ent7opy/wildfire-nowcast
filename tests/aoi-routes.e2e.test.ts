/**
 * End-to-end test for the AOI route handlers: create → list → read → update
 * → upsert rules → archive, all through the actual route handler exports.
 *
 * The PGlite test db is installed via `_setTestDb` so the route handlers'
 * `tryGetDb()` returns it instead of trying to dial Neon.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { makeFreshTestDb } from "@/db/test/pglite";
import { _setTestDb, type AppDb } from "@/lib/db/client";
import type { PGlite } from "@electric-sql/pglite";

import { GET as aoiList, POST as aoiCreate } from "@/app/api/aoi/route";
import {
  GET as aoiOne,
  PATCH as aoiPatch,
  DELETE as aoiDelete,
} from "@/app/api/aoi/[id]/route";
import { PUT as aoiPutRules } from "@/app/api/aoi/[id]/rules/route";

const SONOMA_POLY = {
  type: "Polygon",
  coordinates: [
    [
      [-122.72, 38.42],
      [-122.62, 38.42],
      [-122.62, 38.5],
      [-122.72, 38.5],
      [-122.72, 38.42],
    ],
  ],
};

function jsonRequest(method: string, body?: unknown): Request {
  return new Request("http://localhost/api/aoi", {
    method,
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

describe("AOI routes (Next.js App Router handlers)", () => {
  let pglite: PGlite;
  let db: AppDb;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
    _setTestDb(db);
  });

  afterEach(async () => {
    _setTestDb(null);
    await pglite.close();
  });

  it("walks the full CRUD lifecycle through the route handlers", async () => {
    // 1. POST /api/aoi
    const createReq = jsonRequest("POST", {
      name: "Spring Creek Preserve",
      geometry: SONOMA_POLY,
    });
    const createRes = await aoiCreate(createReq as Parameters<typeof aoiCreate>[0]);
    expect(createRes.status).toBe(201);
    const createBody = (await createRes.json()) as {
      aoi: { id: string; name: string; regionBucket: string };
      rules: { distanceBufferKm: number };
    };
    expect(createBody.aoi.name).toBe("Spring Creek Preserve");
    expect(createBody.aoi.regionBucket).toBe("5x5:W125_N35");
    expect(createBody.rules.distanceBufferKm).toBe(25);

    const aoiId = createBody.aoi.id;

    // 2. GET /api/aoi
    const listRes = await aoiList();
    expect(listRes.status).toBe(200);
    const listBody = (await listRes.json()) as {
      aois: Array<{ id: string; name: string }>;
    };
    expect(listBody.aois).toHaveLength(1);
    expect(listBody.aois[0].id).toBe(aoiId);

    // 3. GET /api/aoi/[id]
    const oneRes = await aoiOne(jsonRequest("GET") as Parameters<typeof aoiOne>[0], {
      params: Promise.resolve({ id: aoiId }),
    });
    expect(oneRes.status).toBe(200);
    const oneBody = (await oneRes.json()) as {
      aoi: { id: string; polygon: { type: string } };
      rules: { distanceBufferKm: number } | null;
    };
    expect(oneBody.aoi.id).toBe(aoiId);
    expect(oneBody.aoi.polygon.type).toBe("MultiPolygon");
    expect(oneBody.rules?.distanceBufferKm).toBe(25);

    // 4. PATCH /api/aoi/[id] — rename
    const patchReq = jsonRequest("PATCH", { name: "Spring Creek (renamed)" });
    const patchRes = await aoiPatch(
      patchReq as Parameters<typeof aoiPatch>[0],
      { params: Promise.resolve({ id: aoiId }) },
    );
    expect(patchRes.status).toBe(200);
    const patchBody = (await patchRes.json()) as { aoi: { name: string } };
    expect(patchBody.aoi.name).toBe("Spring Creek (renamed)");

    // 5. PUT /api/aoi/[id]/rules — replace rules
    const putReq = jsonRequest("PUT", {
      distanceBufferKm: 40,
      minConfidence: "high",
      minFrpMw: 8,
      quietHours: { tz: "America/Los_Angeles", startHour: 22, endHour: 7 },
      notifyChannels: [{ type: "email", target: "ranger@example.org" }],
    });
    const putRes = await aoiPutRules(
      putReq as Parameters<typeof aoiPutRules>[0],
      { params: Promise.resolve({ id: aoiId }) },
    );
    expect(putRes.status).toBe(200);
    const putBody = (await putRes.json()) as {
      rules: { distanceBufferKm: number; notifyChannels: unknown[] };
    };
    expect(putBody.rules.distanceBufferKm).toBe(40);
    expect(putBody.rules.notifyChannels).toHaveLength(1);

    // 6. DELETE /api/aoi/[id] — soft delete
    const delRes = await aoiDelete(
      jsonRequest("DELETE") as Parameters<typeof aoiDelete>[0],
      { params: Promise.resolve({ id: aoiId }) },
    );
    expect(delRes.status).toBe(200);

    // 7. List again — empty.
    const listAfter = await aoiList();
    const listAfterBody = (await listAfter.json()) as { aois: unknown[] };
    expect(listAfterBody.aois).toHaveLength(0);

    // 8. GET on the archived id — 404.
    const oneAfter = await aoiOne(
      jsonRequest("GET") as Parameters<typeof aoiOne>[0],
      { params: Promise.resolve({ id: aoiId }) },
    );
    expect(oneAfter.status).toBe(404);
  });

  it("returns 503 when DATABASE_URL is unset and no test db is installed", async () => {
    _setTestDb(null);
    delete process.env.DATABASE_URL;
    const res = await aoiList();
    expect(res.status).toBe(503);
    const body = (await res.json()) as { error: { code: string } };
    expect(body.error.code).toBe("service_unavailable");
  });

  it("returns 400 with structured details on a malformed body", async () => {
    const badReq = jsonRequest("POST", {
      name: "no geometry here",
    });
    const res = await aoiCreate(badReq as Parameters<typeof aoiCreate>[0]);
    expect(res.status).toBe(400);
    const body = (await res.json()) as {
      error: { code: string; details?: unknown };
    };
    expect(body.error.code).toBe("validation_failed");
    expect(body.error.details).toBeDefined();
  });
});
