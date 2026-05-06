import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { PGlite } from "@electric-sql/pglite";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import {
  clearBriefShareToken,
  getBriefByIdForUser,
  getBriefByShareToken,
  listAoisWithLatestBrief,
  listBriefsForAoi,
  setBriefShareToken,
} from "@/lib/db/aoi-repository";
import { SOFIA, SONOMA, seedAoi, seedBrief, seedUser } from "./_helpers";

describe("dashboard repository reads", () => {
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

  it("listAoisWithLatestBrief returns latest brief timestamp per AOI", async () => {
    const a1 = await seedAoi(db, "user_alice", "preserve-1", SONOMA);
    const a2 = await seedAoi(db, "user_alice", "preserve-2", SOFIA);
    const olderDate = new Date("2026-01-01T00:00:00Z");
    const newerDate = new Date("2026-04-01T00:00:00Z");
    await seedBrief(db, { aoiId: a1, createdAt: olderDate });
    await seedBrief(db, { aoiId: a1, createdAt: newerDate });
    // a2 has no briefs

    const rows = await listAoisWithLatestBrief(db, "user_alice");
    expect(rows).toHaveLength(2);
    const byId = new Map(rows.map((r) => [r.id, r]));
    expect(byId.get(a1)?.lastBriefAt?.toISOString()).toBe(newerDate.toISOString());
    expect(byId.get(a2)?.lastBriefAt).toBeNull();
  });

  it("listAoisWithLatestBrief isolates users", async () => {
    await seedAoi(db, "user_alice", "alice-only", SONOMA);
    const bobAoi = await seedAoi(db, "user_bob", "bob-only", SOFIA);
    await seedBrief(db, { aoiId: bobAoi });

    const aliceRows = await listAoisWithLatestBrief(db, "user_alice");
    expect(aliceRows.map((r) => r.name)).toEqual(["alice-only"]);
  });

  it("listBriefsForAoi honors limit and ownership", async () => {
    const aoiId = await seedAoi(db, "user_alice", "limited", SONOMA);
    for (let i = 0; i < 5; i++) {
      await seedBrief(db, {
        aoiId,
        createdAt: new Date(2026, 0, i + 1),
      });
    }
    const limited = await listBriefsForAoi(db, {
      userId: "user_alice",
      aoiId,
      limit: 3,
    });
    expect(limited).toHaveLength(3);

    const crossUser = await listBriefsForAoi(db, {
      userId: "user_bob",
      aoiId,
    });
    expect(crossUser).toHaveLength(0);
  });

  it("getBriefByIdForUser enforces ownership", async () => {
    const aliceAoi = await seedAoi(db, "user_alice", "shared", SONOMA);
    const briefId = await seedBrief(db, { aoiId: aliceAoi });

    const ok = await getBriefByIdForUser(db, {
      userId: "user_alice",
      briefId,
    });
    expect(ok?.id).toBe(briefId);

    const cross = await getBriefByIdForUser(db, {
      userId: "user_bob",
      briefId,
    });
    expect(cross).toBeNull();
  });

  it("share token: mint → idempotent → public read → expiry → revoke → 404", async () => {
    const aoiId = await seedAoi(db, "user_alice", "shareable", SONOMA);
    const briefId = await seedBrief(db, { aoiId });

    const minted = await setBriefShareToken(db, {
      userId: "user_alice",
      briefId,
      mintToken: () => "tok_abc",
      now: new Date("2026-01-01T00:00:00Z"),
    });
    expect(minted?.token).toBe("tok_abc");

    // Idempotent: second call returns same token even with a different mintFn
    const second = await setBriefShareToken(db, {
      userId: "user_alice",
      briefId,
      mintToken: () => "tok_xyz",
      now: new Date("2026-01-02T00:00:00Z"),
    });
    expect(second?.token).toBe("tok_abc");

    const fetched = await getBriefByShareToken(db, "tok_abc", {
      now: new Date("2026-01-15T00:00:00Z"),
    });
    expect(fetched?.id).toBe(briefId);

    // Past expiry → null
    const expired = await getBriefByShareToken(db, "tok_abc", {
      now: new Date("2026-12-31T00:00:00Z"),
    });
    expect(expired).toBeNull();

    // Unknown token → null
    const missing = await getBriefByShareToken(db, "tok_unknown");
    expect(missing).toBeNull();

    const cleared = await clearBriefShareToken(db, {
      userId: "user_alice",
      briefId,
    });
    expect(cleared).toBe(true);

    const afterClear = await getBriefByShareToken(db, "tok_abc");
    expect(afterClear).toBeNull();
  });

  it("share token: cross-user mint returns null", async () => {
    const aoiId = await seedAoi(db, "user_alice", "alice-priv", SONOMA);
    const briefId = await seedBrief(db, { aoiId });
    const result = await setBriefShareToken(db, {
      userId: "user_bob",
      briefId,
      mintToken: () => "tok_evil",
    });
    expect(result).toBeNull();
  });
});
