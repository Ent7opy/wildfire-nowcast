/**
 * Stage 9 — AOI POST after-response wiring tests.
 *
 * The route uses Next.js `after()` to schedule watch-confirmed dispatch +
 * backfill post-response. In tests we install a synchronous `_setTestAfterImpl`
 * that runs the callback inline so we can assert side effects.
 *
 * Verifies:
 *   - 201 returned with the AOI shape
 *   - watch-confirmed dispatcher invoked with the right args
 *   - backfill invoked with the right args
 *   - watch-confirmed throwing does NOT prevent backfill or break the 201
 *   - backfill throwing does NOT prevent the 201
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { PGlite } from "@electric-sql/pglite";
import { makeFreshTestDb } from "@/db/test/pglite";
import { _setTestDb, type AppDb } from "@/lib/db/client";
import { _setTestAuth } from "@/lib/auth/context";
import {
  POST as aoiCreate,
  _setTestAfterImpl,
  _setTestWatchConfirmed,
  _setTestBackfill,
} from "@/app/api/aoi/route";

const ROUTE_USER_ID = "user_2abcStage9";

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

function jsonReq(body: unknown): Request {
  return new Request("http://localhost/api/aoi", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

describe("AOI POST — after-response wiring (Stage 9)", () => {
  let db: AppDb;
  let pglite: PGlite;
  const queue: Array<() => void | Promise<void>> = [];

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
    _setTestDb(db);
    _setTestAuth(() => ({ ok: true, userId: ROUTE_USER_ID }));
    queue.length = 0;
    _setTestAfterImpl((cb) => {
      queue.push(cb);
    });
  });

  afterEach(async () => {
    _setTestDb(null);
    _setTestAuth(null);
    _setTestAfterImpl(null);
    _setTestWatchConfirmed(null);
    _setTestBackfill(null);
    await pglite.close();
  });

  async function drainQueue(): Promise<void> {
    while (queue.length > 0) {
      const cb = queue.shift()!;
      await cb();
    }
  }

  it("schedules watch-confirmed and backfill via after(); returns 201", async () => {
    const watchCalls: Array<{ aoiId: string; aoiName: string }> = [];
    const backfillCalls: Array<{ aoiId: string; regionBucket: string }> = [];
    _setTestWatchConfirmed(async (_db, args) => {
      watchCalls.push({ aoiId: args.aoiId, aoiName: args.aoiName });
      return { status: "sent", providerMessageId: "fake" };
    });
    _setTestBackfill(async (_db, args) => {
      backfillCalls.push({ aoiId: args.aoiId, regionBucket: args.regionBucket });
      return {
        aoiId: args.aoiId,
        status: "ok",
        detectionsFetched: 0,
        detectionsMatched: 0,
        eventsCreated: 0,
        briefsGenerated: 0,
        notificationsSent: 0,
        durationMs: 1,
      };
    });

    const res = await aoiCreate(
      jsonReq({ name: "Spring Creek Preserve", geometry: SONOMA_POLY }) as Parameters<typeof aoiCreate>[0],
    );
    expect(res.status).toBe(201);
    const body = (await res.json()) as { aoi: { id: string; name: string } };
    expect(body.aoi.name).toBe("Spring Creek Preserve");

    // Queue holds the after-callback; nothing fired yet.
    expect(watchCalls).toHaveLength(0);
    await drainQueue();

    expect(watchCalls).toHaveLength(1);
    expect(watchCalls[0].aoiId).toBe(body.aoi.id);
    expect(watchCalls[0].aoiName).toBe("Spring Creek Preserve");
    expect(backfillCalls).toHaveLength(1);
    expect(backfillCalls[0].aoiId).toBe(body.aoi.id);
    expect(backfillCalls[0].regionBucket).toBe("5x5:W125_N35");
  });

  it("watch-confirmed throwing does not prevent backfill nor break the 201", async () => {
    let backfillCalled = false;
    _setTestWatchConfirmed(async () => {
      throw new Error("simulated email outage");
    });
    _setTestBackfill(async (_db, args) => {
      backfillCalled = true;
      return {
        aoiId: args.aoiId,
        status: "ok",
        detectionsFetched: 0,
        detectionsMatched: 0,
        eventsCreated: 0,
        briefsGenerated: 0,
        notificationsSent: 0,
        durationMs: 1,
      };
    });

    const res = await aoiCreate(
      jsonReq({ name: "Test", geometry: SONOMA_POLY }) as Parameters<typeof aoiCreate>[0],
    );
    expect(res.status).toBe(201);
    await drainQueue();
    expect(backfillCalled).toBe(true);
  });

  it("backfill throwing does not break the 201", async () => {
    _setTestWatchConfirmed(async () => ({
      status: "sent",
      providerMessageId: "x",
    }));
    _setTestBackfill(async () => {
      throw new Error("simulated firms outage");
    });

    const res = await aoiCreate(
      jsonReq({ name: "Test", geometry: SONOMA_POLY }) as Parameters<typeof aoiCreate>[0],
    );
    expect(res.status).toBe(201);
    await drainQueue();
    // No throw — both AOI creation and 201 succeed; the after-task swallowed
    // the backfill error per the brief's failure-isolation rule.
  });
});
