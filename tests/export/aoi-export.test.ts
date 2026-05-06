import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { PGlite } from "@electric-sql/pglite";
import { NextRequest } from "next/server";
import { makeFreshTestDb } from "@/db/test/pglite";
import { _setTestDb, type AppDb } from "@/lib/db/client";
import { _setTestAuth } from "@/lib/auth/context";
import { GET as aoiExport } from "@/app/api/aoi/[id]/export/route";
import { GET as portfolioGeojson } from "@/app/api/export/aois.geojson/route";
import { GET as portfolioCsv } from "@/app/api/export/briefs.csv/route";
import { POSITIONING_LINE } from "@/lib/export/positioning";
import { seedAoi, seedBrief, seedUser } from "../dashboard/_helpers";

const ALICE = "user_alice_export";
const BOB = "user_bob_export";

describe("export routes (PGlite)", () => {
  let db: AppDb;
  let pglite: PGlite;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
    _setTestDb(db);
    await seedUser(db, ALICE);
    await seedUser(db, BOB);
    _setTestAuth(() => ({ ok: true, userId: ALICE }));
  });
  afterEach(async () => {
    _setTestDb(null);
    _setTestAuth(null);
    await pglite.close();
  });

  it("per-AOI geojson export: Feature shape, properties.rules, ownership", async () => {
    const aoiId = await seedAoi(db, ALICE, "shape-test");
    const req = new NextRequest(
      `http://test/api/aoi/${aoiId}/export?format=geojson`,
    );
    const res = await aoiExport(req, { params: Promise.resolve({ id: aoiId }) });
    expect(res.status).toBe(200);
    expect(res.headers.get("Content-Type")).toContain("application/geo+json");
    const body = (await res.json()) as {
      type: string;
      geometry: { type: string };
      properties: {
        name: string;
        rules: { distanceBufferKm: number };
      };
    };
    expect(body.type).toBe("Feature");
    expect(body.geometry.type).toBe("MultiPolygon");
    expect(body.properties.name).toBe("shape-test");
    expect(body.properties.rules.distanceBufferKm).toBe(25);

    // Cross-user: Bob requests Alice's AOI
    _setTestAuth(() => ({ ok: true, userId: BOB }));
    const reqBob = new NextRequest(
      `http://test/api/aoi/${aoiId}/export?format=geojson`,
    );
    const resBob = await aoiExport(reqBob, {
      params: Promise.resolve({ id: aoiId }),
    });
    expect(resBob.status).toBe(404);
  });

  it("per-AOI markdown export: reverse-chron + positioning footer + dashboard link", async () => {
    const aoiId = await seedAoi(db, ALICE, "md-test");
    await seedBrief(db, {
      aoiId,
      createdAt: new Date("2026-01-01T00:00:00Z"),
      summary: "older",
    });
    await seedBrief(db, {
      aoiId,
      createdAt: new Date("2026-04-01T00:00:00Z"),
      summary: "newer",
    });
    const req = new NextRequest(
      `http://test/api/aoi/${aoiId}/export?format=markdown`,
    );
    const res = await aoiExport(req, { params: Promise.resolve({ id: aoiId }) });
    expect(res.status).toBe(200);
    expect(res.headers.get("Content-Type")).toContain("text/markdown");
    const text = await res.text();
    expect(text).toContain(POSITIONING_LINE);
    expect(text).toContain(`/dashboard/aoi/${aoiId}`);
    const newerIdx = text.indexOf("2026-04-01");
    const olderIdx = text.indexOf("2026-01-01");
    expect(newerIdx).toBeGreaterThan(-1);
    expect(olderIdx).toBeGreaterThan(-1);
    expect(newerIdx).toBeLessThan(olderIdx);
  });

  it("portfolio GeoJSON FeatureCollection only includes the user's AOIs", async () => {
    await seedAoi(db, ALICE, "alice-1");
    await seedAoi(db, BOB, "bob-1");
    const res = await portfolioGeojson();
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      type: string;
      features: Array<{ properties: { name: string } }>;
    };
    expect(body.type).toBe("FeatureCollection");
    expect(body.features.map((f) => f.properties.name)).toEqual(["alice-1"]);
  });

  it("portfolio CSV: header + escaping + since filter", async () => {
    const aoiId = await seedAoi(db, ALICE, "csv-test");
    await seedBrief(db, {
      aoiId,
      createdAt: new Date("2025-12-01T00:00:00Z"),
      summary: 'has "quote" and , comma',
    });
    await seedBrief(db, {
      aoiId,
      createdAt: new Date("2026-04-01T00:00:00Z"),
      summary: "recent one",
    });

    const req = new NextRequest("http://test/api/export/briefs.csv");
    const res = await portfolioCsv(req);
    expect(res.status).toBe(200);
    expect(res.headers.get("Content-Type")).toContain("text/csv");
    const text = await res.text();
    const lines = text.trim().split("\n");
    expect(lines[0]).toBe(
      "brief_id,aoi_id,aoi_name,created_at,gate_reason,model,latency_ms,cost_usd_est,last_notified_at,summary",
    );
    expect(text).toContain('"has ""quote"" and , comma"');

    // since filter
    const reqSince = new NextRequest(
      "http://test/api/export/briefs.csv?since=2026-01-01",
    );
    const resSince = await portfolioCsv(reqSince);
    const textSince = await resSince.text();
    const linesSince = textSince.trim().split("\n");
    expect(linesSince).toHaveLength(2); // header + 1 row
    expect(textSince).toContain("recent one");
    expect(textSince).not.toContain("has ");
  });
});
