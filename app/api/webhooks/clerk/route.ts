/**
 * Clerk → Wildfire Nowcast webhook receiver.
 *
 * Receives Svix-signed `user.created` / `user.updated` / `user.deleted`
 * events from Clerk and syncs the local `users` table. The signature is
 * verified against `CLERK_WEBHOOK_SIGNING_SECRET` per Clerk's docs:
 * https://clerk.com/docs/integrations/webhooks/sync-data
 *
 * Build-without-blocking: when the signing secret is unset, return 503
 * `service_unavailable` and log a warning. Never process unsigned payloads.
 *
 * The raw body is consumed via `req.text()` because Svix verifies over the
 * raw bytes; calling `req.json()` first would corrupt the verification.
 */
import { NextResponse, type NextRequest } from "next/server";
import { sql } from "drizzle-orm";
import { Webhook } from "svix";
import { tryGetDb, type AppDb } from "@/lib/db/client";

export const runtime = "nodejs";

type ClerkUserEvent = {
  type: "user.created" | "user.updated" | "user.deleted";
  data: {
    id: string;
    email_addresses?: Array<{
      id: string;
      email_address: string;
    }>;
    primary_email_address_id?: string | null;
    first_name?: string | null;
    last_name?: string | null;
    deleted?: boolean;
  };
};

type Verifier = (payload: string, headers: Record<string, string>) => unknown;

export type WebhookDeps = {
  /** Override Svix verification (tests). */
  verify?: Verifier;
};

export async function POST(req: NextRequest): Promise<NextResponse> {
  return handle(req, {});
}

export async function _handleForTest(
  req: NextRequest,
  deps: WebhookDeps,
): Promise<NextResponse> {
  return handle(req, deps);
}

async function handle(
  req: NextRequest,
  deps: WebhookDeps,
): Promise<NextResponse> {
  const secret = process.env.CLERK_WEBHOOK_SIGNING_SECRET;
  if (!secret && !deps.verify) {
    console.warn(
      "[clerk-webhook] CLERK_WEBHOOK_SIGNING_SECRET unset; refusing payload.",
    );
    return NextResponse.json(
      {
        error: {
          code: "service_unavailable",
          message: "CLERK_WEBHOOK_SIGNING_SECRET is not configured",
        },
      },
      { status: 503 },
    );
  }

  const db = tryGetDb();
  if (!db) {
    return NextResponse.json(
      {
        error: {
          code: "service_unavailable",
          message: "DATABASE_URL is not configured",
        },
      },
      { status: 503 },
    );
  }

  const rawBody = await req.text();
  const headers: Record<string, string> = {
    "svix-id": req.headers.get("svix-id") ?? "",
    "svix-timestamp": req.headers.get("svix-timestamp") ?? "",
    "svix-signature": req.headers.get("svix-signature") ?? "",
  };

  let evt: ClerkUserEvent;
  try {
    const verify =
      deps.verify ??
      ((payload: string, h: Record<string, string>) =>
        new Webhook(secret as string).verify(payload, h));
    evt = verify(rawBody, headers) as ClerkUserEvent;
  } catch (err) {
    console.warn(
      `[clerk-webhook] signature verification failed: ${
        err instanceof Error ? err.message : String(err)
      }`,
    );
    return NextResponse.json(
      {
        error: {
          code: "unauthenticated",
          message: "Invalid Svix signature",
        },
      },
      { status: 401 },
    );
  }

  switch (evt.type) {
    case "user.created":
    case "user.updated":
      await upsertUser(db, evt);
      return NextResponse.json({ ok: true });
    case "user.deleted":
      await softDeleteUser(db, evt.data.id);
      return NextResponse.json({ ok: true });
    default:
      console.warn(
        `[clerk-webhook] unknown event type: ${(evt as { type: string }).type}`,
      );
      return NextResponse.json(
        {
          error: {
            code: "validation_failed",
            message: `Unsupported event type: ${(evt as { type: string }).type}`,
          },
        },
        { status: 400 },
      );
  }
}

function pickPrimaryEmail(evt: ClerkUserEvent): string {
  const list = evt.data.email_addresses ?? [];
  const primaryId = evt.data.primary_email_address_id;
  if (primaryId) {
    const found = list.find((e) => e.id === primaryId);
    if (found) return found.email_address;
  }
  if (list.length > 0) return list[0].email_address;
  // Fallback: keep the row syncable — webhooks can briefly arrive before the
  // primary email is confirmed. Placeholder is replaced by the next
  // `user.updated` event.
  return `${evt.data.id}@pending.invalid`;
}

function pickDisplayName(evt: ClerkUserEvent): string | null {
  const first = evt.data.first_name?.trim();
  const last = evt.data.last_name?.trim();
  if (!first && !last) return null;
  return [first, last].filter(Boolean).join(" ");
}

async function upsertUser(db: AppDb, evt: ClerkUserEvent): Promise<void> {
  const email = pickPrimaryEmail(evt);
  const displayName = pickDisplayName(evt);
  await db.execute(sql`
    INSERT INTO "users" ("id", "email", "display_name")
    VALUES (${evt.data.id}, ${email}, ${displayName})
    ON CONFLICT ("id") DO UPDATE SET
      "email" = EXCLUDED."email",
      "display_name" = EXCLUDED."display_name",
      "deleted_at" = NULL
  `);
}

async function softDeleteUser(db: AppDb, userId: string): Promise<void> {
  await db.execute(sql`
    UPDATE "users" SET "deleted_at" = now() WHERE "id" = ${userId}
  `);
}
