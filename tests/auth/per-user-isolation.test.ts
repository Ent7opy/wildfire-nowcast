/**
 * Stage 5 — per-user AOI isolation at the repository layer.
 *
 * Two Clerk-style users each create one AOI; assert each can only see and
 * modify their own. Covers the cross-user existence check that prevents
 * `archiveAoi` from leaking 500s.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import {
  archiveAoi,
  AoiNotFoundError,
  createAoi,
  getAoiById,
  listAois,
} from "@/lib/db/aoi-repository";
import type { AppDb } from "@/lib/db/client";
import type { PGlite } from "@electric-sql/pglite";
import type { PolygonalGeom } from "@/lib/validators/geojson";

const POLY_A: PolygonalGeom = {
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

const POLY_B: PolygonalGeom = {
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

const USER_A = "user_2abcAlice";
const USER_B = "user_2abcBob";

describe("AOI repository — per-user isolation", () => {
  let db: AppDb;
  let pglite: PGlite;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
    await db.execute(sql`
      INSERT INTO "users" ("id", "email") VALUES (${USER_A}, ${"alice@example.org"}),
                                                  (${USER_B}, ${"bob@example.org"})
    `);
  });
  afterEach(async () => {
    await pglite.close();
  });

  it("listAois returns only the caller's AOIs", async () => {
    await createAoi(db, { userId: USER_A, name: "Alice AOI", geometry: POLY_A });
    await createAoi(db, { userId: USER_B, name: "Bob AOI", geometry: POLY_B });

    const aliceList = await listAois(db, USER_A);
    expect(aliceList).toHaveLength(1);
    expect(aliceList[0].name).toBe("Alice AOI");

    const bobList = await listAois(db, USER_B);
    expect(bobList).toHaveLength(1);
    expect(bobList[0].name).toBe("Bob AOI");
  });

  it("getAoiById returns null when fetched as the other user", async () => {
    const bob = await createAoi(db, { userId: USER_B, name: "Bob AOI", geometry: POLY_B });
    const fromAlice = await getAoiById(db, USER_A, bob.aoi.id);
    expect(fromAlice).toBeNull();
  });

  it("archiveAoi for another user's AOI throws AoiNotFoundError (not 500)", async () => {
    const bob = await createAoi(db, { userId: USER_B, name: "Bob AOI", geometry: POLY_B });
    await expect(
      archiveAoi(db, { userId: USER_A, aoiId: bob.aoi.id }),
    ).rejects.toBeInstanceOf(AoiNotFoundError);
    // And Bob's AOI is still active.
    const stillThere = await getAoiById(db, USER_B, bob.aoi.id);
    expect(stillThere).not.toBeNull();
  });
});
