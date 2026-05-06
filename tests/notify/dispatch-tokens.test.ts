/**
 * Stage 7 — verifies the dispatcher mints four action tokens per outbound
 * email and re-dispatch (idempotency-skip) does not mint more.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import type { AppDb } from "@/lib/db/client";
import { dispatchBrief } from "@/lib/notify/dispatch";
import type { SendResult } from "@/lib/notify/resend";
import type { PGlite } from "@electric-sql/pglite";

async function seed(db: AppDb): Promise<{ briefId: string }> {
  const userId = `u-${Math.random().toString(36).slice(2, 8)}`;
  await db.execute(sql`INSERT INTO "users" (id, email) VALUES (${userId}, 'fallback@x.org')`);
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
            ${JSON.stringify([{ type: "email", target: "alice@x.org" }])}::jsonb)
  `);
  const ev = (await db.execute(sql`
    INSERT INTO "aoi_events" (aoi_id, first_seen_at, last_seen_at, nearest_distance_km,
      detection_count, dedupe_hash, status)
    VALUES (${aoiId}, '2026-04-21T00:00:00Z', '2026-04-21T01:00:00Z', 1, 1, 'h', 'new')
    RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  const eventId = ((ev.rows ?? ev) as Array<{ id: string }>)[0].id;
  const brief = (await db.execute(sql`
    INSERT INTO "aoi_briefs" (aoi_id, event_id, model, gate_reason, payload, rendered_markdown)
    VALUES (${aoiId}, ${eventId}, 'test', 'multi_pixel',
            ${JSON.stringify({ summary: "x" })}::jsonb, '# brief\n\nbody')
    RETURNING id
  `)) as unknown as { rows?: Array<{ id: string }> };
  return { briefId: ((brief.rows ?? brief) as Array<{ id: string }>)[0].id };
}

async function countTokens(db: AppDb, briefId: string): Promise<number> {
  const r = (await db.execute(sql`
    SELECT count(*)::int AS n FROM "notify_action_tokens" WHERE "brief_id" = ${briefId}
  `)) as unknown as { rows?: Array<{ n: number }> };
  return Number(((r.rows ?? r) as Array<{ n: number }>)[0].n);
}

describe("dispatchBrief — token minting", () => {
  let db: AppDb;
  let pglite: PGlite;
  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
  });
  afterEach(async () => {
    await pglite.close();
  });

  it("first dispatch mints four action tokens and renders them in the email markdown", async () => {
    const { briefId } = await seed(db);
    const sent: Array<{ to: string; markdown: string }> = [];
    const send = async (a: { to: string; markdown: string }): Promise<SendResult> => {
      sent.push({ to: a.to, markdown: a.markdown });
      return { ok: true, providerMessageId: "x", latencyMs: 1 };
    };
    await dispatchBrief(db, briefId, { send });
    expect(sent).toHaveLength(1);
    expect(await countTokens(db, briefId)).toBe(4);
    expect(sent[0].markdown).toContain("/api/notify/snooze/");
    expect(sent[0].markdown).toContain("/api/notify/pause/");
    expect(sent[0].markdown).toContain("/api/notify/unsubscribe/");
    expect(sent[0].markdown).toContain("/api/notify/feedback/");
  });

  it("re-dispatch (skipped/duplicate) does not mint more tokens", async () => {
    const { briefId } = await seed(db);
    const send = async (): Promise<SendResult> => ({ ok: true, providerMessageId: "x", latencyMs: 1 });
    await dispatchBrief(db, briefId, { send });
    const after1 = await countTokens(db, briefId);
    await dispatchBrief(db, briefId, { send });
    const after2 = await countTokens(db, briefId);
    expect(after1).toBe(4);
    expect(after2).toBe(4);
  });
});
