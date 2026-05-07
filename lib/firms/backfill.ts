/**
 * Stage 9 — first-AOI backfill.
 *
 * Single entry point: `backfillForNewAoi(db, args)`. Called immediately after
 * AOI creation (post-response via `next/server`'s `after`) to give a user who
 * arrives mid-fire an immediate brief instead of waiting up to 15 minutes for
 * the next cron tick.
 *
 * Behaviour:
 *   1. Open a `job_runs` row with `job_name='aoi-backfill'` so cron-poll
 *      latency metrics aren't contaminated.
 *   2. Skip cleanly when `FIRMS_MAP_KEY` is unset (build-without-blocking).
 *   3. Fetch last 24h of FIRMS detections for the new AOI's bucket.
 *   4. Run the matcher scoped to JUST this AOI.
 *   5. Pipe matches through the brief generator + Stage 4 dispatcher.
 *
 * Failures must NOT propagate — the AOI POST returns 201 either way.
 */
import { sql } from "drizzle-orm";
import type { AppDb } from "@/lib/db/client";
import {
  fetchAreaCsv,
  type FirmsBbox,
  type FirmsFetchResult,
  type FirmsSource,
} from "./client";
import { bucketToBbox } from "./buckets";
import { matchDetectionsToAois } from "./matcher";
import {
  generateBriefForEvent,
  type GenerateOutcome,
} from "@/lib/ai/generate";
import { dispatchBrief, type DispatchOutcome } from "@/lib/notify/dispatch";

export type BackfillOutcome = {
  aoiId: string;
  status: "ok" | "skipped" | "error";
  reason?: "config_missing" | "fetch_failed";
  detectionsFetched: number;
  detectionsMatched: number;
  eventsCreated: number;
  briefsGenerated: number;
  notificationsSent: number;
  durationMs: number;
};

export type FirmsFetchFn = (args: {
  source: FirmsSource;
  bbox: FirmsBbox;
  dayRange?: number;
}) => Promise<FirmsFetchResult>;

export type BackfillArgs = {
  aoiId: string;
  userId: string;
  regionBucket: string;
  source?: FirmsSource;
  now?: Date;
  fetchImpl?: FirmsFetchFn;
  briefGen?: typeof generateBriefForEvent;
  notifyDispatch?: typeof dispatchBrief;
};

const DEFAULT_SOURCE: FirmsSource = "VIIRS_NOAA20_NRT";

export async function backfillForNewAoi(
  db: AppDb,
  args: BackfillArgs,
): Promise<BackfillOutcome> {
  const start = Date.now();
  const source = args.source ?? DEFAULT_SOURCE;
  const runId = await openJobRun(db, args.regionBucket, start);

  if (!process.env.FIRMS_MAP_KEY && !args.fetchImpl) {
    await closeJobRun(db, runId, {
      status: "ok",
      finishedAt: new Date(),
    });
    return {
      aoiId: args.aoiId,
      status: "skipped",
      reason: "config_missing",
      detectionsFetched: 0,
      detectionsMatched: 0,
      eventsCreated: 0,
      briefsGenerated: 0,
      notificationsSent: 0,
      durationMs: Date.now() - start,
    };
  }

  let bbox: FirmsBbox;
  try {
    bbox = bucketToBbox(args.regionBucket);
  } catch (err) {
    await closeJobRun(db, runId, {
      status: "error",
      error: errMessage(err),
      finishedAt: new Date(),
    });
    return {
      aoiId: args.aoiId,
      status: "error",
      reason: "fetch_failed",
      detectionsFetched: 0,
      detectionsMatched: 0,
      eventsCreated: 0,
      briefsGenerated: 0,
      notificationsSent: 0,
      durationMs: Date.now() - start,
    };
  }

  const fetchFn = args.fetchImpl ?? fetchAreaCsv;
  const fetchResult = await fetchFn({ source, bbox, dayRange: 1 });
  if (!fetchResult.ok) {
    await closeJobRun(db, runId, {
      status: "error",
      error: `${fetchResult.code}: ${fetchResult.message}`,
      firmsRequestCount: 1,
      finishedAt: new Date(),
    });
    return {
      aoiId: args.aoiId,
      status: "error",
      reason: "fetch_failed",
      detectionsFetched: 0,
      detectionsMatched: 0,
      eventsCreated: 0,
      briefsGenerated: 0,
      notificationsSent: 0,
      durationMs: Date.now() - start,
    };
  }

  const matchOutcome = await matchDetectionsToAois(db, {
    bucket: args.regionBucket,
    source,
    detections: fetchResult.detections,
    aoiIds: [args.aoiId],
  });

  let briefsGenerated = 0;
  const generatedBriefIds: string[] = [];
  const briefGen = args.briefGen ?? generateBriefForEvent;
  for (const eventId of matchOutcome.createdEventIds) {
    let outcome: GenerateOutcome;
    try {
      outcome = await briefGen(db, eventId);
    } catch (err) {
      outcome = { status: "error", eventId, reason: errMessage(err) };
    }
    if (outcome.status === "generated") {
      briefsGenerated += 1;
      if (outcome.briefId) generatedBriefIds.push(outcome.briefId);
    }
  }

  let notificationsSent = 0;
  const dispatcher = args.notifyDispatch ?? dispatchBrief;
  for (const briefId of generatedBriefIds) {
    let outcome: DispatchOutcome;
    try {
      outcome = await dispatcher(db, briefId);
    } catch {
      continue;
    }
    for (const a of outcome.attempts) {
      if (a.status === "sent") notificationsSent += 1;
    }
  }

  await closeJobRun(db, runId, {
    status: "ok",
    firmsRequestCount: 1,
    detectionsInserted: matchOutcome.detectionsInserted,
    eventsCreated: matchOutcome.eventsCreated,
    briefsGenerated,
    notificationsSent,
    finishedAt: new Date(),
  });

  return {
    aoiId: args.aoiId,
    status: "ok",
    detectionsFetched: fetchResult.detections.length,
    detectionsMatched: matchOutcome.detectionsInserted,
    eventsCreated: matchOutcome.eventsCreated,
    briefsGenerated,
    notificationsSent,
    durationMs: Date.now() - start,
  };
}

// ---------------------------------------------------------------------------
// job_runs helpers
//
// Kept local to avoid coupling the cron route's helpers; the SQL is small and
// duplicate INSERT/UPDATE trivially safe.

async function openJobRun(
  db: AppDb,
  bucket: string,
  startedAtMs: number,
): Promise<string> {
  const startedAt = new Date(startedAtMs).toISOString();
  const result = (await db.execute(sql`
    INSERT INTO "job_runs" ("job_name", "bucket", "started_at", "status")
    VALUES ('aoi-backfill', ${bucket}, ${startedAt}, 'running')
    RETURNING "id"
  `)) as unknown as { rows?: Array<{ id: string | number | bigint }> };
  const rows = (result.rows ?? (result as unknown as Array<{ id: string | number | bigint }>)) as Array<{
    id: string | number | bigint;
  }>;
  return String(rows[0]?.id ?? "");
}

type CloseArgs = {
  status: "ok" | "error";
  error?: string;
  finishedAt: Date;
  firmsRequestCount?: number;
  detectionsInserted?: number;
  eventsCreated?: number;
  briefsGenerated?: number;
  notificationsSent?: number;
};

async function closeJobRun(
  db: AppDb,
  id: string,
  args: CloseArgs,
): Promise<void> {
  if (!id) return;
  await db.execute(sql`
    UPDATE "job_runs"
    SET
      "finished_at" = ${args.finishedAt.toISOString()},
      "status" = ${args.status},
      "error" = ${args.error ?? null},
      "firms_request_count" = COALESCE("firms_request_count", 0) + ${args.firmsRequestCount ?? 0},
      "detections_inserted" = COALESCE("detections_inserted", 0) + ${args.detectionsInserted ?? 0},
      "events_created" = COALESCE("events_created", 0) + ${args.eventsCreated ?? 0},
      "briefs_generated" = COALESCE("briefs_generated", 0) + ${args.briefsGenerated ?? 0},
      "notifications_sent" = COALESCE("notifications_sent", 0) + ${args.notificationsSent ?? 0}
    WHERE "id" = ${id}
  `);
}

function errMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}
