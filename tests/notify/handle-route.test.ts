/**
 * Coverage for `app/api/notify/_lib/handle.ts` failure paths.
 *
 * Token IS the auth on these routes, so the failure modes (unknown / wrong-
 * action / expired) are security-critical: each MUST return the same opaque
 * 404 so the four endpoints can't be used as an oracle for which (action,
 * channel, target) a guessed token is bound to. PR #427 landed the opaque-404
 * fix; this file pins it down. Also covers the snooze + pause happy paths
 * end-to-end through the route — `applyX` helpers are unit-tested already in
 * `actions.test.ts`, but no test wires the route → handler → DB chain.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import {
  loadActionToken,
  mintActionToken,
  type ActionKind,
} from "@/lib/notify/action-tokens";
import type { PGlite } from "@electric-sql/pglite";
import type { NextRequest } from "next/server";

let currentDb: AppDb | null = null;

vi.mock("@/lib/db/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/db/client")>();
  return { ...actual, tryGetDb: () => currentDb };
});

async function seedAoi(db: AppDb): Promise<{ aoiId: string }> {
  const userId = `u-${Math.random().toString(36).slice(2, 8)}`;
  await db.execute(
    sql`INSERT INTO "users" (id, email) VALUES (${userId}, ${userId + "@x"})`,
  );
  const polygon = JSON.stringify({
    type: "Polygon",
    coordinates: [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
  });
  const aoi = (await db.execute(sql`
    INSERT INTO "aois" (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
    VALUES (${userId}, 'A', ${polygon}, ${polygon},
            ${JSON.stringify({ type: "Point", coordinates: [0.5, 0.5] })},
            '5x5:E000_N00', 100) RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const aoiId = ((aoi.rows ?? aoi) as Array<{ id: string }>)[0].id;
  await db.execute(sql`
    INSERT INTO "aoi_rules" (aoi_id, distance_buffer_km, min_confidence, min_frp_mw, notify_channels)
    VALUES (${aoiId}, 25, 'nominal', 5,
            ${JSON.stringify([{ type: "email", target: "alice@example.org" }])}::jsonb)
  `);
  return { aoiId };
}

async function mintFor(db: AppDb, aoiId: string, action: ActionKind): Promise<string> {
  const m = await mintActionToken(db, {
    aoiId, briefId: null, action, channel: "email", target: "alice@example.org",
  });
  return m.token;
}

async function readPaused(db: AppDb, aoiId: string): Promise<Date | null> {
  const r = (await db.execute(sql`
    SELECT "paused_until" FROM "aoi_rules" WHERE "aoi_id" = ${aoiId}
  `)) as unknown as { rows?: Array<{ paused_until: Date | string | null }> };
  const v = ((r.rows ?? r) as Array<{ paused_until: Date | string | null }>)[0]?.paused_until ?? null;
  return v == null ? null : v instanceof Date ? v : new Date(v);
}

const mkReq = (url: string): NextRequest => new Request(url) as unknown as NextRequest;

describe("notify handle — opaque 404 contract", () => {
  let db: AppDb;
  let pglite: PGlite;
  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
    currentDb = db;
  });
  afterEach(async () => {
    currentDb = null;
    await pglite.close();
  });

  it("cross-action attack: snooze token at /pause is rejected without side effects", async () => {
    const { aoiId } = await seedAoi(db);
    const token = await mintFor(db, aoiId, "snooze");
    const { GET } = await import("@/app/api/notify/pause/[token]/route");
    const res = await GET(
      mkReq(`http://localhost/api/notify/pause/${token}`),
      { params: Promise.resolve({ token }) },
    );
    expect(res.status).toBe(404);
    expect(await res.text()).toContain("Link not found");

    // The rejected click MUST NOT consume the token (else a passive scanner
    // mis-routing the URL could permanently burn the user's real link), and
    // MUST NOT trigger the wrong handler's side effect.
    const loaded = await loadActionToken(db, token);
    expect(loaded?.redeemedAt).toBeNull();
    expect(await readPaused(db, aoiId)).toBeNull();
  });

  it("expired token returns the same opaque body as an unknown token", async () => {
    const { aoiId } = await seedAoi(db);
    const token = await mintFor(db, aoiId, "snooze");
    await db.execute(sql`
      UPDATE "notify_action_tokens" SET "expires_at" = '2020-01-01T00:00:00Z'
      WHERE "token" = ${token}
    `);
    const { GET } = await import("@/app/api/notify/snooze/[token]/route");

    const expiredRes = await GET(
      mkReq(`http://localhost/api/notify/snooze/${token}`),
      { params: Promise.resolve({ token }) },
    );
    const fakeToken = "deadbeef".repeat(8);
    const unknownRes = await GET(
      mkReq(`http://localhost/api/notify/snooze/${fakeToken}`),
      { params: Promise.resolve({ token: fakeToken }) },
    );

    expect(expiredRes.status).toBe(404);
    expect(unknownRes.status).toBe(404);
    // Byte-identical bodies — the whole point of PR #427. If these ever
    // diverge, an attacker can distinguish a guessed-real-but-expired token
    // from a guessed-fake one and cheaply confirm token validity.
    expect(await expiredRes.text()).toBe(await unknownRes.text());

    expect((await loadActionToken(db, token))?.redeemedAt).toBeNull();
  });
});

describe("notify handle — non-feedback happy paths via the route", () => {
  let db: AppDb;
  let pglite: PGlite;
  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
    currentDb = db;
  });
  afterEach(async () => {
    currentDb = null;
    await pglite.close();
  });

  it("snooze: GET succeeds, redeems the token, sets paused_until ~24h out", async () => {
    const { aoiId } = await seedAoi(db);
    const token = await mintFor(db, aoiId, "snooze");
    const before = Date.now();
    const { GET } = await import("@/app/api/notify/snooze/[token]/route");
    const res = await GET(
      mkReq(`http://localhost/api/notify/snooze/${token}`),
      { params: Promise.resolve({ token }) },
    );
    expect(res.status).toBe(200);
    expect(await res.text()).toContain("Snoozed for 24 hours");
    expect((await loadActionToken(db, token))?.redeemedAt).not.toBeNull();
    const paused = await readPaused(db, aoiId);
    expect(paused).not.toBeNull();
    const delta = paused!.getTime() - before;
    expect(delta).toBeGreaterThan(23 * 3600_000);
    expect(delta).toBeLessThan(25 * 3600_000);
  });

  it("pause: GET succeeds and sets paused_until far in the future", async () => {
    const { aoiId } = await seedAoi(db);
    const token = await mintFor(db, aoiId, "pause");
    const { GET } = await import("@/app/api/notify/pause/[token]/route");
    const res = await GET(
      mkReq(`http://localhost/api/notify/pause/${token}`),
      { params: Promise.resolve({ token }) },
    );
    expect(res.status).toBe(200);
    expect(await res.text()).toContain("Paused");
    const paused = await readPaused(db, aoiId);
    expect(paused).not.toBeNull();
    expect(paused!.getTime() - Date.now()).toBeGreaterThan(50 * 365 * 86400_000);
  });
});
