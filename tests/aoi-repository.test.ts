import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { makeFreshTestDb } from "@/db/test/pglite";
import {
  archiveAoi,
  AoiAreaTooLargeError,
  AoiNameConflictError,
  AoiNotFoundError,
  createAoi,
  getAoiById,
  getRulesByAoiId,
  listAois,
  updateAoi,
  upsertRules,
} from "@/lib/db/aoi-repository";
import { STUB_USER_ID } from "@/db/schema";
import type { AppDb } from "@/lib/db/client";
import type { PGlite } from "@electric-sql/pglite";

import type { PolygonalGeom } from "@/lib/validators/geojson";

const SONOMA_POLY: PolygonalGeom = {
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

const SOFIA_POLY: PolygonalGeom = {
  type: "Polygon",
  coordinates: [
    [
      [23.3, 42.65],
      [23.4, 42.65],
      [23.4, 42.75],
      [23.3, 42.75],
      [23.3, 42.65],
    ],
  ],
};

const HUGE_POLY: PolygonalGeom = {
  type: "Polygon",
  coordinates: [
    [
      [0, 0],
      [10, 0],
      [10, 10],
      [0, 10],
      [0, 0],
    ],
  ],
};

describe("aoi repository (PGlite)", () => {
  let db: AppDb;
  let pglite: PGlite;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
  });

  afterEach(async () => {
    await pglite.close();
  });

  it("creates → reads → lists → updates → archives an AOI", async () => {
    const created = await createAoi(db, {
      userId: STUB_USER_ID,
      name: "Spring Creek Preserve",
      geometry: SONOMA_POLY,
    });
    expect(created.aoi.regionBucket).toBe("5x5:W125_N35");
    expect(created.aoi.areaHa).toBeGreaterThan(0);
    expect(created.rules.distanceBufferKm).toBe(25);

    const fetched = await getAoiById(db, STUB_USER_ID, created.aoi.id);
    expect(fetched?.name).toBe("Spring Creek Preserve");
    expect(fetched?.polygon.type).toBe("MultiPolygon");

    const list = await listAois(db, STUB_USER_ID);
    expect(list).toHaveLength(1);
    expect(list[0].id).toBe(created.aoi.id);

    const updated = await updateAoi(db, {
      userId: STUB_USER_ID,
      aoiId: created.aoi.id,
      patch: { name: "Spring Creek Preserve (north unit)" },
    });
    expect(updated.name).toBe("Spring Creek Preserve (north unit)");

    const reGeom = await updateAoi(db, {
      userId: STUB_USER_ID,
      aoiId: created.aoi.id,
      patch: { geometry: SOFIA_POLY },
    });
    expect(reGeom.regionBucket).toBe("5x5:E020_N40");

    await archiveAoi(db, { userId: STUB_USER_ID, aoiId: created.aoi.id });
    const afterArchive = await listAois(db, STUB_USER_ID);
    expect(afterArchive).toHaveLength(0);
    const fetchedAfter = await getAoiById(db, STUB_USER_ID, created.aoi.id);
    expect(fetchedAfter).toBeNull();
  });

  it("rejects polygons over the 100,000 ha cap", async () => {
    // ~10° × 10° square ≈ 1.2M km² ≈ 120M ha
    await expect(
      createAoi(db, { userId: STUB_USER_ID, name: "way too big", geometry: HUGE_POLY }),
    ).rejects.toBeInstanceOf(AoiAreaTooLargeError);
  });

  it("conflicts on a duplicate active name for the same user", async () => {
    await createAoi(db, {
      userId: STUB_USER_ID,
      name: "Same",
      geometry: SONOMA_POLY,
    });
    await expect(
      createAoi(db, {
        userId: STUB_USER_ID,
        name: "Same",
        geometry: SOFIA_POLY,
      }),
    ).rejects.toBeInstanceOf(AoiNameConflictError);
  });

  it("permits reusing a name once the prior AOI is archived", async () => {
    const first = await createAoi(db, {
      userId: STUB_USER_ID,
      name: "Reusable",
      geometry: SONOMA_POLY,
    });
    await archiveAoi(db, { userId: STUB_USER_ID, aoiId: first.aoi.id });
    const second = await createAoi(db, {
      userId: STUB_USER_ID,
      name: "Reusable",
      geometry: SOFIA_POLY,
    });
    expect(second.aoi.id).not.toBe(first.aoi.id);
  });

  it("update on a non-existent AOI throws AoiNotFoundError", async () => {
    await expect(
      updateAoi(db, {
        userId: STUB_USER_ID,
        aoiId: "00000000-0000-0000-0000-000000000000",
        patch: { name: "x" },
      }),
    ).rejects.toBeInstanceOf(AoiNotFoundError);
  });

  it("rules upsert: create then update in place", async () => {
    const created = await createAoi(db, {
      userId: STUB_USER_ID,
      name: "Rules Test",
      geometry: SONOMA_POLY,
    });

    const first = await upsertRules(db, {
      userId: STUB_USER_ID,
      aoiId: created.aoi.id,
      rules: {
        distanceBufferKm: 50,
        minConfidence: "high",
        minFrpMw: 10,
        quietHours: { tz: "America/Los_Angeles", startHour: 22, endHour: 7 },
        pausedUntil: null,
        notifyChannels: [{ type: "email", target: "ranger@example.org" }],
      },
    });
    expect(first.distanceBufferKm).toBe(50);
    expect(first.minConfidence).toBe("high");
    expect(first.notifyChannels).toHaveLength(1);

    const second = await upsertRules(db, {
      userId: STUB_USER_ID,
      aoiId: created.aoi.id,
      rules: {
        distanceBufferKm: 15,
        minConfidence: "low",
        minFrpMw: 1,
        quietHours: null,
        pausedUntil: null,
        notifyChannels: [],
      },
    });
    expect(second.distanceBufferKm).toBe(15);
    expect(second.notifyChannels).toEqual([]);

    const fromGet = await getRulesByAoiId(db, created.aoi.id);
    expect(fromGet?.minConfidence).toBe("low");
  });

  it("rules upsert against a non-existent AOI throws AoiNotFoundError", async () => {
    await expect(
      upsertRules(db, {
        userId: STUB_USER_ID,
        aoiId: "00000000-0000-0000-0000-000000000000",
        rules: {
          distanceBufferKm: 25,
          minConfidence: "nominal",
          minFrpMw: 5,
          quietHours: null,
          pausedUntil: null,
          notifyChannels: [],
        },
      }),
    ).rejects.toBeInstanceOf(AoiNotFoundError);
  });
});
