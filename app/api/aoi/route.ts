/**
 * /api/aoi — list + create AOIs.
 *
 * Spec: docs/SPEC-A-prime-v1.md §API surface.
 * Auth: Clerk per-user via `withDb`.
 * Runtime: Node.js (default) — pg driver is not Edge-compatible.
 *
 * Stage 9: after the 201 returns, two best-effort tasks fire via Next.js
 * `after()` (the App Router post-response primitive):
 *   1. `dispatchWatchConfirmed` — one-shot "now watching ..." email.
 *   2. `backfillForNewAoi` — synchronous backfill over last 24h FIRMS for
 *      this AOI's bucket. Failures here MUST NOT break AOI creation.
 *
 * Backfill-path decision (per brief 23 §"Open questions"): we chose Path A
 * (`after` from `next/server`). It's the documented Next.js / Vercel
 * primitive for "do this after responding," requires no new route, and
 * preserves the single AOI-creation transaction. If Vercel logs ever show
 * function termination before `after`'s callback resolves, switch to Path C
 * (internal `/api/aoi/[id]/backfill` self-call) per the brief.
 */
import { NextResponse, type NextRequest } from "next/server";
import { after } from "next/server";
import { aoiCreateSchema } from "@/lib/validators/aoi";
import { createAoi, listAois } from "@/lib/db/aoi-repository";
import { parseJson, withDb } from "@/lib/api/handlers";
import {
  dispatchWatchConfirmed,
  absoluteAoiUrl,
} from "@/lib/notify/watch-confirmed";
import { backfillForNewAoi } from "@/lib/firms/backfill";
import type { AppDb } from "@/lib/db/client";

// Hobby max function duration is 60s. Most of the backfill happens after the
// 201 returns, but the function instance must stay alive until `after`'s
// callback resolves — so we still want the 60s ceiling.
export const maxDuration = 60;
export const runtime = "nodejs";

// Test-only injection points so PGlite tests can exercise the after-response
// path synchronously and stub out network deps.
type AfterFn = (cb: () => void | Promise<void>) => void;
let testAfterImpl: AfterFn | null = null;
let testWatchConfirmed: typeof dispatchWatchConfirmed | null = null;
let testBackfill: typeof backfillForNewAoi | null = null;

export function _setTestAfterImpl(fn: AfterFn | null): void {
  testAfterImpl = fn;
}
export function _setTestWatchConfirmed(
  fn: typeof dispatchWatchConfirmed | null,
): void {
  testWatchConfirmed = fn;
}
export function _setTestBackfill(
  fn: typeof backfillForNewAoi | null,
): void {
  testBackfill = fn;
}

export async function GET(): Promise<NextResponse> {
  return withDb(async ({ db, userId }) => {
    const rows = await listAois(db, userId);
    return NextResponse.json({
      aois: rows.map((r) => ({
        id: r.id,
        name: r.name,
        regionBucket: r.regionBucket,
        areaHa: r.areaHa,
        createdAt: r.createdAt.toISOString(),
      })),
    });
  });
}

export async function POST(req: NextRequest): Promise<NextResponse> {
  const parsed = await parseJson(req, aoiCreateSchema);
  if (!parsed.ok) return parsed.response;
  return withDb(async ({ db, userId }) => {
    const { aoi, rules } = await createAoi(db, {
      userId,
      name: parsed.value.name,
      geometry: parsed.value.geometry,
    });

    scheduleAfterResponse(db, {
      aoiId: aoi.id,
      userId,
      aoiName: aoi.name,
      regionBucket: aoi.regionBucket,
      areaHa: aoi.areaHa,
      createdAt: aoi.createdAt,
    });

    return NextResponse.json(
      {
        aoi: {
          id: aoi.id,
          name: aoi.name,
          regionBucket: aoi.regionBucket,
          areaHa: aoi.areaHa,
          createdAt: aoi.createdAt.toISOString(),
          polygon: aoi.polygon,
          bbox: aoi.bbox,
          centroid: aoi.centroid,
        },
        rules: {
          distanceBufferKm: rules.distanceBufferKm,
          minConfidence: rules.minConfidence,
          minFrpMw: rules.minFrpMw,
          quietHours: rules.quietHours,
          pausedUntil: rules.pausedUntil?.toISOString() ?? null,
          notifyChannels: rules.notifyChannels,
        },
      },
      { status: 201 },
    );
  });
}

const FIFTEEN_MIN_MS = 15 * 60 * 1000;

function scheduleAfterResponse(
  db: AppDb,
  args: {
    aoiId: string;
    userId: string;
    aoiName: string;
    regionBucket: string;
    areaHa: number;
    createdAt: Date;
  },
): void {
  const watchConfirmed = testWatchConfirmed ?? dispatchWatchConfirmed;
  const backfill = testBackfill ?? backfillForNewAoi;
  const firstPollAt = new Date(args.createdAt.getTime() + FIFTEEN_MIN_MS);

  const work = async (): Promise<void> => {
    try {
      await watchConfirmed(db, {
        aoiId: args.aoiId,
        userId: args.userId,
        aoiName: args.aoiName,
        regionBucket: args.regionBucket,
        areaHa: args.areaHa,
        firstPollAt,
        aoiUrl: absoluteAoiUrl(args.aoiId),
      });
    } catch (err) {
      console.warn(`[aoi.post] watch-confirmed dispatch failed: ${errMessage(err)}`);
    }
    try {
      await backfill(db, {
        aoiId: args.aoiId,
        userId: args.userId,
        regionBucket: args.regionBucket,
      });
    } catch (err) {
      console.warn(`[aoi.post] backfill failed: ${errMessage(err)}`);
    }
  };

  if (testAfterImpl) {
    testAfterImpl(work);
    return;
  }
  try {
    after(work);
  } catch (err) {
    // `after` throws when called outside a request scope (e.g. in unit tests
    // that exercise the route handler directly). Fall back to fire-and-forget
    // so the wiring still runs; production always has a request scope.
    console.warn(`[aoi.post] after() unavailable; running inline: ${errMessage(err)}`);
    void work();
  }
}

function errMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}
