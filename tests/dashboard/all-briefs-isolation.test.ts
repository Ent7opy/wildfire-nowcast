/**
 * Repository-layer per-user isolation for the cross-AOI brief listings.
 *
 * `listAllBriefsWithPayloadForUser` powers `/api/export/briefs.csv` and
 * `listBriefsForAoiWithPayload` powers `/api/aoi/[id]/export?format=markdown`.
 * The existing route tests in `tests/export/aoi-export.test.ts` only seed
 * briefs for one user, so cross-user leakage would not be detected there.
 * These tests pin the userId filter at the repo layer where the SQL lives.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import type { PGlite } from "@electric-sql/pglite";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import {
  listAllBriefsWithPayloadForUser,
  listBriefsForAoiWithPayload,
} from "@/lib/db/aoi-repository";
import { SOFIA, SONOMA, seedAoi, seedBrief, seedUser } from "./_helpers";

describe("cross-AOI brief listings — per-user isolation", () => {
  let db: AppDb;
  let pglite: PGlite;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
    await seedUser(db, "user_alice");
    await seedUser(db, "user_bob");
  });
  afterEach(async () => {
    await pglite.close();
  });

  it("listAllBriefsWithPayloadForUser only returns the caller's briefs", async () => {
    const aliceAoi = await seedAoi(db, "user_alice", "alice-aoi", SONOMA);
    const bobAoi = await seedAoi(db, "user_bob", "bob-aoi", SOFIA);
    await seedBrief(db, {
      aoiId: aliceAoi,
      createdAt: new Date("2026-04-01T00:00:00Z"),
      summary: "alice-secret",
    });
    await seedBrief(db, {
      aoiId: bobAoi,
      createdAt: new Date("2026-04-02T00:00:00Z"),
      summary: "bob-secret",
    });

    const aliceRows = await listAllBriefsWithPayloadForUser(db, {
      userId: "user_alice",
    });
    expect(aliceRows).toHaveLength(1);
    expect(aliceRows[0].aoiId).toBe(aliceAoi);
    expect(aliceRows[0].aoiName).toBe("alice-aoi");

    const bobRows = await listAllBriefsWithPayloadForUser(db, {
      userId: "user_bob",
    });
    expect(bobRows).toHaveLength(1);
    expect(bobRows[0].aoiId).toBe(bobAoi);
  });

  it("listAllBriefsWithPayloadForUser excludes briefs whose AOI is archived", async () => {
    // Important: the SELECT joins archived_at IS NULL on aois, so archiving
    // an AOI must hide its historical briefs from the export endpoint.
    const aoi = await seedAoi(db, "user_alice", "to-archive", SONOMA);
    await seedBrief(db, { aoiId: aoi, createdAt: new Date("2026-04-01T00:00:00Z") });
    await db.execute(
      sql`UPDATE "aois" SET "archived_at" = NOW() WHERE "id" = ${aoi}`,
    );
    const rows = await listAllBriefsWithPayloadForUser(db, {
      userId: "user_alice",
    });
    expect(rows).toEqual([]);
  });

  it("listBriefsForAoiWithPayload returns no rows when the AOI belongs to another user", async () => {
    const aliceAoi = await seedAoi(db, "user_alice", "alice-private", SONOMA);
    await seedBrief(db, { aoiId: aliceAoi });

    // Bob asks for Alice's AOI by id — repo must return empty, not throw.
    const rows = await listBriefsForAoiWithPayload(db, {
      userId: "user_bob",
      aoiId: aliceAoi,
    });
    expect(rows).toEqual([]);
  });
});
