import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { PGlite } from "@electric-sql/pglite";
import { NextRequest } from "next/server";
import { makeFreshTestDb } from "@/db/test/pglite";
import { _setTestDb, type AppDb } from "@/lib/db/client";
import { _setTestAuth } from "@/lib/auth/context";
import { _setMintTokenForTest } from "@/lib/share/token";
import { POST as sharePost, DELETE as shareDelete } from "@/app/api/brief/[id]/share/route";
import { getBriefByShareToken } from "@/lib/db/aoi-repository";
import { seedAoi, seedBrief, seedUser } from "./_helpers";

const ALICE = "user_alice_share";

describe("share route mint + revoke", () => {
  let db: AppDb;
  let pglite: PGlite;
  let briefId: string;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
    _setTestDb(db);
    await seedUser(db, ALICE);
    _setTestAuth(() => ({ ok: true, userId: ALICE }));
    const aoiId = await seedAoi(db, ALICE, "share-target");
    briefId = await seedBrief(db, { aoiId });
    _setMintTokenForTest(() => "tok_deterministic");
  });
  afterEach(async () => {
    _setTestDb(null);
    _setTestAuth(null);
    _setMintTokenForTest(null);
    await pglite.close();
  });

  it("POST mints a token; second POST is idempotent", async () => {
    const req = new NextRequest(`http://test/api/brief/${briefId}/share`, {
      method: "POST",
    });
    const res = await sharePost(req, { params: Promise.resolve({ id: briefId }) });
    expect(res.status).toBe(200);
    const body = (await res.json()) as { token: string; publicUrl: string };
    expect(body.token).toBe("tok_deterministic");
    expect(body.publicUrl).toContain("/brief/share/tok_deterministic");

    // Second mint is idempotent (returns existing token even with different mintFn)
    _setMintTokenForTest(() => "tok_should_not_be_used");
    const res2 = await sharePost(req, {
      params: Promise.resolve({ id: briefId }),
    });
    const body2 = (await res2.json()) as { token: string };
    expect(body2.token).toBe("tok_deterministic");
  });

  it("public read works after mint, fails after revoke", async () => {
    const req = new NextRequest(`http://test/api/brief/${briefId}/share`, {
      method: "POST",
    });
    await sharePost(req, { params: Promise.resolve({ id: briefId }) });

    const fetched = await getBriefByShareToken(db, "tok_deterministic");
    expect(fetched?.id).toBe(briefId);

    const delReq = new NextRequest(`http://test/api/brief/${briefId}/share`, {
      method: "DELETE",
    });
    const delRes = await shareDelete(delReq, {
      params: Promise.resolve({ id: briefId }),
    });
    expect(delRes.status).toBe(200);

    const afterRevoke = await getBriefByShareToken(db, "tok_deterministic");
    expect(afterRevoke).toBeNull();
  });

  it("non-existent brief → 404", async () => {
    const req = new NextRequest("http://test/api/brief/x/share", {
      method: "POST",
    });
    const res = await sharePost(req, {
      params: Promise.resolve({ id: "00000000-0000-0000-0000-000000000000" }),
    });
    expect(res.status).toBe(404);
  });
});
