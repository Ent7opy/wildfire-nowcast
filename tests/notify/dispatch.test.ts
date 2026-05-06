/**
 * Stage 4 dispatcher unit tests on PGlite.
 *
 * Covers every dispatcher branch named in the brief:
 *   - happy path: one email channel → one `sent` row, last_notified_at set
 *   - empty notify_channels → fallback to user email
 *   - duplicate brief on second invocation → skipped/duplicate, no second row
 *   - paused_until in future → skipped/paused
 *   - quiet-hours window matches → skipped/quiet_hours
 *   - webhook channel → skipped/channel_not_implemented
 *   - send returns config_missing → row written, no last_notified_at update
 *   - send returns provider_error → row written status=failed, no throw
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import { dispatchBrief } from "@/lib/notify/dispatch";
import type { SendResult } from "@/lib/notify/resend";
import type { PGlite } from "@electric-sql/pglite";

type SeedOpts = {
  notifyChannels?: Array<
    | { type: "email"; target: string }
    | { type: "webhook"; target: string }
  >;
  pausedUntil?: Date | null;
  quietHours?: { tz: string; startHour: number; endHour: number } | null;
  userEmail?: string;
  briefSummary?: string;
};

async function seed(db: AppDb, opts: SeedOpts = {}): Promise<{ aoiId: string; briefId: string }> {
  const userId = `stub-user-${Math.random().toString(36).slice(2, 8)}`;
  const userEmail = opts.userEmail ?? "owner@example.org";
  await db.execute(sql`
    INSERT INTO "users" (id, email) VALUES (${userId}, ${userEmail})
  `);

  const polygon = JSON.stringify({
    type: "Polygon",
    coordinates: [
      [[-122.7, 38.4], [-122.6, 38.4], [-122.6, 38.5], [-122.7, 38.5], [-122.7, 38.4]],
    ],
  });
  const aoiRes = (await db.execute(sql`
    INSERT INTO "aois" (user_id, name, polygon, bbox, centroid, region_bucket, area_ha)
    VALUES (
      ${userId}, 'Test Preserve', ${polygon}, ${polygon},
      ${JSON.stringify({ type: "Point", coordinates: [-122.65, 38.45] })},
      '5x5:W125_N35', 100
    ) RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const aoiRows = (aoiRes.rows ?? (aoiRes as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  const aoiId = aoiRows[0].id;

  const channels = opts.notifyChannels ?? [];
  await db.execute(sql`
    INSERT INTO "aoi_rules" (aoi_id, distance_buffer_km, min_confidence, min_frp_mw, paused_until, quiet_hours, notify_channels)
    VALUES (
      ${aoiId}, 25, 'nominal', 5,
      ${opts.pausedUntil ? opts.pausedUntil.toISOString() : null},
      ${opts.quietHours ? JSON.stringify(opts.quietHours) : null}::jsonb,
      ${JSON.stringify(channels)}::jsonb
    )
  `);

  const evRes = (await db.execute(sql`
    INSERT INTO "aoi_events" (
      aoi_id, first_seen_at, last_seen_at, nearest_distance_km,
      detection_count, peak_frp_mw, dedupe_hash, status
    ) VALUES (
      ${aoiId}, '2026-04-21T04:00:00Z', '2026-04-21T04:30:00Z',
      8, 2, 11, ${"hash-" + Math.random().toString(36).slice(2, 10)}, 'new'
    ) RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const evRows = (evRes.rows ?? (evRes as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  const eventId = evRows[0].id;

  const summary = opts.briefSummary ?? "Spring Creek Preserve fire ~8 km NE of boundary; FRP ~11 MW";
  const payload = {
    schema_version: 1,
    aoi: { id: aoiId, name: "Test Preserve", area_ha: 100 },
    summary,
    key_facts: {
      nearest_detection_km: 8,
      bearing_from_aoi_deg: 90,
      wind_dir_deg: null,
      wind_speed_kmh: null,
      wind_toward_aoi: null,
      detection_count_in_window: 2,
      max_frp_mw: 11,
      satellites: ["VIIRS_NOAA20_NRT"],
      window_hours: 24,
    },
    context: {
      weather_note: null,
      authority_perimeter: { source: null, posted_ts: null, contains_detection: null },
      prior_events: [],
    },
    recommended_watch_items: ["item"],
    uncertainty: "fixture",
    next_brief_hint: { when: "next tick", trigger: "fixture" },
  };
  const briefRes = (await db.execute(sql`
    INSERT INTO "aoi_briefs" (
      aoi_id, event_id, model, gate_reason, payload, rendered_markdown
    ) VALUES (
      ${aoiId}, ${eventId}, 'test/stub', 'multi_pixel',
      ${JSON.stringify(payload)}::jsonb,
      ${"# Test Preserve — situation brief\n\n" + summary + "\n"}
    ) RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const briefRows = (briefRes.rows ?? (briefRes as unknown as Array<{ id: string }>)) as Array<{ id: string }>;
  return { aoiId, briefId: briefRows[0].id };
}

async function readNotifications(db: AppDb, briefId: string): Promise<Array<Record<string, unknown>>> {
  const r = (await db.execute(sql`
    SELECT * FROM "notifications_log" WHERE "brief_id" = ${briefId} ORDER BY "sent_at" ASC, "status" ASC
  `)) as unknown as { rows?: Array<Record<string, unknown>> };
  return (r.rows ?? (r as unknown as Array<Record<string, unknown>>)) as Array<Record<string, unknown>>;
}

async function readBriefLastNotified(db: AppDb, briefId: string): Promise<Date | string | null> {
  const r = (await db.execute(sql`
    SELECT "last_notified_at" FROM "aoi_briefs" WHERE "id" = ${briefId}
  `)) as unknown as { rows?: Array<{ last_notified_at: Date | string | null }> };
  const rows = (r.rows ?? (r as unknown as Array<{ last_notified_at: Date | string | null }>)) as Array<{
    last_notified_at: Date | string | null;
  }>;
  return rows[0]?.last_notified_at ?? null;
}

describe("dispatchBrief — PGlite", () => {
  let db: AppDb;
  let pglite: PGlite;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
  });

  afterEach(async () => {
    await pglite.close();
  });

  it("happy path — explicit email channel sends and stamps last_notified_at", async () => {
    const { briefId } = await seed(db, {
      notifyChannels: [{ type: "email", target: "alice@example.org" }],
    });
    const send = async (): Promise<SendResult> => ({
      ok: true,
      providerMessageId: "resend-1",
      latencyMs: 12,
    });
    const outcome = await dispatchBrief(db, briefId, { send });
    expect(outcome.attempts).toHaveLength(1);
    expect(outcome.attempts[0].status).toBe("sent");
    const rows = await readNotifications(db, briefId);
    expect(rows).toHaveLength(1);
    expect(rows[0].status).toBe("sent");
    expect(rows[0].provider_message_id).toBe("resend-1");
    expect(await readBriefLastNotified(db, briefId)).not.toBeNull();
  });

  it("falls back to users.email when notify_channels is empty", async () => {
    const { briefId } = await seed(db, {
      notifyChannels: [],
      userEmail: "fallback@example.org",
    });
    const calls: Array<{ to: string }> = [];
    const send = async (a: { to: string }): Promise<SendResult> => {
      calls.push({ to: a.to });
      return { ok: true, providerMessageId: "fallback-1", latencyMs: 1 };
    };
    const outcome = await dispatchBrief(db, briefId, { send });
    expect(calls[0].to).toBe("fallback@example.org");
    expect(outcome.attempts[0].status).toBe("sent");
  });

  it("second invocation is skipped/duplicate without writing a second row", async () => {
    const { briefId } = await seed(db, {
      notifyChannels: [{ type: "email", target: "dup@example.org" }],
    });
    const send = async (): Promise<SendResult> => ({
      ok: true,
      providerMessageId: "x",
      latencyMs: 1,
    });
    await dispatchBrief(db, briefId, { send });
    const second = await dispatchBrief(db, briefId, { send });
    expect(second.attempts[0].status).toBe("skipped");
    if (second.attempts[0].status === "skipped") {
      expect(second.attempts[0].reason).toBe("duplicate");
    }
    const rows = await readNotifications(db, briefId);
    expect(rows).toHaveLength(1);
  });

  it("paused_until in future → skipped/paused", async () => {
    const { briefId } = await seed(db, {
      notifyChannels: [{ type: "email", target: "p@example.org" }],
      pausedUntil: new Date(Date.now() + 60 * 60 * 1000),
    });
    let called = false;
    const send = async (): Promise<SendResult> => {
      called = true;
      return { ok: true, providerMessageId: "x", latencyMs: 1 };
    };
    const outcome = await dispatchBrief(db, briefId, { send });
    expect(called).toBe(false);
    expect(outcome.attempts[0].status).toBe("skipped");
    if (outcome.attempts[0].status === "skipped") {
      expect(outcome.attempts[0].reason).toBe("paused");
    }
    const rows = await readNotifications(db, briefId);
    expect(rows[0].skip_reason).toBe("paused");
  });

  it("quiet hours window matches → skipped/quiet_hours", async () => {
    const { briefId } = await seed(db, {
      notifyChannels: [{ type: "email", target: "q@example.org" }],
      quietHours: { tz: "America/Los_Angeles", startHour: 0, endHour: 23 },
    });
    // 12:00 UTC == 04:00 or 05:00 LA depending on DST, both inside [0,23).
    const now = new Date("2026-04-21T12:00:00Z");
    const outcome = await dispatchBrief(db, briefId, {
      now,
      send: async () => ({ ok: true, providerMessageId: "x", latencyMs: 1 }),
    });
    expect(outcome.attempts[0].status).toBe("skipped");
    if (outcome.attempts[0].status === "skipped") {
      expect(outcome.attempts[0].reason).toBe("quiet_hours");
    }
  });

  it("webhook channel is recorded as skipped/channel_not_implemented", async () => {
    const { briefId } = await seed(db, {
      notifyChannels: [{ type: "webhook", target: "https://example.org/hook" }],
    });
    let called = false;
    const send = async (): Promise<SendResult> => {
      called = true;
      return { ok: true, providerMessageId: "x", latencyMs: 1 };
    };
    const outcome = await dispatchBrief(db, briefId, { send });
    expect(called).toBe(false);
    expect(outcome.attempts).toHaveLength(1);
    expect(outcome.attempts[0].status).toBe("skipped");
    const rows = await readNotifications(db, briefId);
    expect(rows[0].skip_reason).toBe("channel_not_implemented");
    expect(rows[0].channel).toBe("webhook");
  });

  it("send returns config_missing → row written; no last_notified_at update", async () => {
    const { briefId } = await seed(db, {
      notifyChannels: [{ type: "email", target: "x@example.org" }],
    });
    const send = async (): Promise<SendResult> => ({
      ok: false,
      code: "config_missing",
      message: "no key",
      latencyMs: 0,
    });
    const outcome = await dispatchBrief(db, briefId, { send });
    expect(outcome.attempts[0].status).toBe("config_missing");
    const rows = await readNotifications(db, briefId);
    expect(rows[0].status).toBe("config_missing");
    expect(await readBriefLastNotified(db, briefId)).toBeNull();
  });

  it("user email is @pending.invalid placeholder → skipped/no_recipient_pending, send not called", async () => {
    const { briefId } = await seed(db, {
      notifyChannels: [],
      userEmail: "clerk_user_xyz@pending.invalid",
    });
    let called = false;
    const send = async (): Promise<SendResult> => {
      called = true;
      return { ok: true, providerMessageId: "x", latencyMs: 1 };
    };
    const outcome = await dispatchBrief(db, briefId, { send });
    expect(called).toBe(false);
    expect(outcome.attempts).toHaveLength(1);
    expect(outcome.attempts[0].status).toBe("skipped");
    if (outcome.attempts[0].status === "skipped") {
      expect(outcome.attempts[0].reason).toBe("no_recipient_pending");
    }
    const rows = await readNotifications(db, briefId);
    expect(rows).toHaveLength(1);
    expect(rows[0].status).toBe("skipped");
    expect(rows[0].skip_reason).toBe("no_recipient_pending");
  });

  it("missing user (deleted) → skipped/no_recipient row persisted", async () => {
    const { briefId, aoiId } = await seed(db, {
      notifyChannels: [],
      userEmail: "owner@example.org",
    });
    // Detach the user so the LEFT JOIN yields null user_email. The aois.user_id
    // FK has ON DELETE CASCADE; drop whichever auto-named FK is in place so we
    // can dangle the reference without cascading away the AOI.
    const fk = (await db.execute(sql`
      SELECT conname FROM pg_constraint
      WHERE conrelid = 'aois'::regclass AND contype = 'f'
        AND pg_get_constraintdef(oid) LIKE '%REFERENCES%users%'
    `)) as unknown as { rows?: Array<{ conname: string }> };
    const fkRows = (fk.rows ?? (fk as unknown as Array<{ conname: string }>)) as Array<{
      conname: string;
    }>;
    for (const r of fkRows) {
      await db.execute(sql.raw(`ALTER TABLE "aois" DROP CONSTRAINT "${r.conname}"`));
    }
    await db.execute(sql`UPDATE "aois" SET "user_id" = ${"missing-user-" + Math.random().toString(36).slice(2, 8)} WHERE "id" = ${aoiId}`);
    let called = false;
    const send = async (): Promise<SendResult> => {
      called = true;
      return { ok: true, providerMessageId: "x", latencyMs: 1 };
    };
    const outcome = await dispatchBrief(db, briefId, { send });
    expect(called).toBe(false);
    expect(outcome.attempts).toHaveLength(1);
    expect(outcome.attempts[0].status).toBe("skipped");
    if (outcome.attempts[0].status === "skipped") {
      expect(outcome.attempts[0].reason).toBe("no_recipient");
    }
    const rows = await readNotifications(db, briefId);
    expect(rows).toHaveLength(1);
    expect(rows[0].status).toBe("skipped");
    expect(rows[0].skip_reason).toBe("no_recipient");
  });

  it("webhook channel re-dispatch → second call returns skipped/duplicate", async () => {
    const { briefId } = await seed(db, {
      notifyChannels: [{ type: "webhook", target: "https://example.org/hook" }],
    });
    const send = async (): Promise<SendResult> => ({
      ok: true,
      providerMessageId: "x",
      latencyMs: 1,
    });
    const first = await dispatchBrief(db, briefId, { send });
    expect(first.attempts[0].status).toBe("skipped");
    if (first.attempts[0].status === "skipped") {
      expect(first.attempts[0].reason).toBe("channel_not_implemented");
    }
    const second = await dispatchBrief(db, briefId, { send });
    expect(second.attempts[0].status).toBe("skipped");
    if (second.attempts[0].status === "skipped") {
      expect(second.attempts[0].reason).toBe("duplicate");
    }
    const rows = await readNotifications(db, briefId);
    expect(rows).toHaveLength(1);
    expect(rows[0].skip_reason).toBe("channel_not_implemented");
  });

  it("send returns provider_error → row written status=failed; dispatcher does not throw", async () => {
    const { briefId } = await seed(db, {
      notifyChannels: [{ type: "email", target: "x@example.org" }],
    });
    const send = async (): Promise<SendResult> => ({
      ok: false,
      code: "provider_error",
      message: "boom",
      latencyMs: 0,
    });
    const outcome = await dispatchBrief(db, briefId, { send });
    expect(outcome.attempts[0].status).toBe("failed");
    const rows = await readNotifications(db, briefId);
    expect(rows[0].status).toBe("failed");
    expect(String(rows[0].error)).toContain("provider_error");
    expect(await readBriefLastNotified(db, briefId)).toBeNull();
  });
});
