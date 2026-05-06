/**
 * Stage 5 — Clerk webhook handler branch coverage.
 *
 * The Svix verifier is injected via the test-only `_handleForTest` hook so
 * tests don't need a real signing secret; the production POST handler still
 * uses the live `Webhook(secret).verify` path.
 */
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { sql } from "drizzle-orm";
import { makeFreshTestDb } from "@/db/test/pglite";
import { _setTestDb, type AppDb } from "@/lib/db/client";
import type { PGlite } from "@electric-sql/pglite";
import { POST as webhookPost, _handleForTest } from "@/app/api/webhooks/clerk/route";

const SAVED_ENV = { ...process.env };

function makeReq(body: unknown): Request {
  return new Request("http://localhost/api/webhooks/clerk", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "svix-id": "msg_test",
      "svix-timestamp": String(Math.floor(Date.now() / 1000)),
      "svix-signature": "v1,test",
    },
    body: JSON.stringify(body),
  });
}

describe("Clerk webhook receiver", () => {
  let db: AppDb;
  let pglite: PGlite;

  beforeEach(async () => {
    ({ db, pglite } = await makeFreshTestDb());
    _setTestDb(db);
    process.env.CLERK_WEBHOOK_SIGNING_SECRET = "whsec_test";
  });
  afterEach(async () => {
    _setTestDb(null);
    await pglite.close();
    process.env = { ...SAVED_ENV };
  });

  it("upserts a user on user.created", async () => {
    const evt = {
      type: "user.created",
      data: {
        id: "user_2abcCreated",
        email_addresses: [
          { id: "idem_1", email_address: "created@example.org" },
        ],
        primary_email_address_id: "idem_1",
        first_name: "Cre",
        last_name: "Ated",
      },
    };
    const res = await _handleForTest(makeReq(evt) as Parameters<typeof _handleForTest>[0], { verify: () => evt });
    expect(res.status).toBe(200);
    const rows = (await db.execute(sql`
      SELECT id, email, display_name FROM users WHERE id = 'user_2abcCreated'
    `)) as unknown as { rows?: Array<{ id: string; email: string; display_name: string | null }> };
    const r = (rows.rows ?? (rows as unknown as Array<{ id: string; email: string; display_name: string | null }>)) as Array<{
      id: string;
      email: string;
      display_name: string | null;
    }>;
    expect(r[0].email).toBe("created@example.org");
    expect(r[0].display_name).toBe("Cre Ated");
  });

  it("updates existing row on user.updated", async () => {
    await db.execute(sql`
      INSERT INTO users (id, email) VALUES ('user_2abcUpdated', 'old@example.org')
    `);
    const evt = {
      type: "user.updated",
      data: {
        id: "user_2abcUpdated",
        email_addresses: [
          { id: "idem_2", email_address: "new@example.org" },
        ],
        primary_email_address_id: "idem_2",
        first_name: "New",
        last_name: null,
      },
    };
    const res = await _handleForTest(makeReq(evt) as Parameters<typeof _handleForTest>[0], { verify: () => evt });
    expect(res.status).toBe(200);
    const rows = (await db.execute(sql`
      SELECT email FROM users WHERE id = 'user_2abcUpdated'
    `)) as unknown as { rows?: Array<{ email: string }> };
    const r = (rows.rows ?? (rows as unknown as Array<{ email: string }>)) as Array<{ email: string }>;
    expect(r[0].email).toBe("new@example.org");
  });

  it("soft-deletes on user.deleted", async () => {
    await db.execute(sql`
      INSERT INTO users (id, email) VALUES ('user_2abcDeleted', 'd@example.org')
    `);
    const evt = {
      type: "user.deleted",
      data: { id: "user_2abcDeleted" },
    };
    const res = await _handleForTest(makeReq(evt) as Parameters<typeof _handleForTest>[0], { verify: () => evt });
    expect(res.status).toBe(200);
    const rows = (await db.execute(sql`
      SELECT deleted_at FROM users WHERE id = 'user_2abcDeleted'
    `)) as unknown as { rows?: Array<{ deleted_at: string | Date | null }> };
    const r = (rows.rows ?? (rows as unknown as Array<{ deleted_at: string | Date | null }>)) as Array<{
      deleted_at: string | Date | null;
    }>;
    expect(r[0].deleted_at).not.toBeNull();
  });

  it("returns 401 on bad signature", async () => {
    const evt = { type: "user.created", data: { id: "user_2bad" } };
    const res = await _handleForTest(makeReq(evt) as Parameters<typeof _handleForTest>[0], {
      verify: () => {
        throw new Error("bad signature");
      },
    });
    expect(res.status).toBe(401);
    const body = (await res.json()) as { error: { code: string } };
    expect(body.error.code).toBe("unauthenticated");
  });

  it("returns 503 when CLERK_WEBHOOK_SIGNING_SECRET is unset and no test verify", async () => {
    delete process.env.CLERK_WEBHOOK_SIGNING_SECRET;
    const res = await webhookPost(makeReq({ type: "user.created", data: { id: "x" } }) as Parameters<typeof webhookPost>[0]);
    expect(res.status).toBe(503);
  });

  it("returns 400 on unknown event type", async () => {
    const evt = { type: "session.created", data: { id: "sess_x" } };
    const res = await _handleForTest(makeReq(evt) as Parameters<typeof _handleForTest>[0], { verify: () => evt });
    expect(res.status).toBe(400);
  });
});
