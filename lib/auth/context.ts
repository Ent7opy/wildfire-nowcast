/**
 * Stage 5 — Clerk auth context.
 *
 * `requireUserId()` is the single seam every authenticated route hits via
 * `withDb`. It hides the Clerk SDK from callers and applies the
 * build-without-blocking discipline: when `CLERK_SECRET_KEY` is unset, return
 * a typed `config_missing` rather than throwing at module load.
 *
 * Tests inject a synchronous resolver via `_setTestAuth` so they don't need
 * Clerk's runtime. Production leaves it null and we call Clerk's `auth()`.
 *
 * No DB writes happen here; user-row provisioning lives in `ensureUserExists`
 * (called from `withDb` after a positive auth) and in the Clerk webhook.
 */
import { sql } from "drizzle-orm";
import type { AppDb } from "@/lib/db/client";

export type AuthErrorCode = "unauthenticated" | "config_missing";

export type AuthResult =
  | { ok: true; userId: string }
  | { ok: false; code: AuthErrorCode };

export type TestAuthFn = () => Promise<AuthResult> | AuthResult;

let testAuth: TestAuthFn | null = null;

/**
 * Test-only injection point. Production passes `null`; the unit suite passes
 * a function returning the desired `AuthResult`. Mirrors `_setTestFirmsFetch`
 * in `app/api/aoi/poll/route.ts`.
 */
export function _setTestAuth(fn: TestAuthFn | null): void {
  testAuth = fn;
}

export async function requireUserId(): Promise<AuthResult> {
  if (testAuth) return await testAuth();

  if (!process.env.CLERK_SECRET_KEY) {
    return { ok: false, code: "config_missing" };
  }

  // Lazy import so the module graph doesn't pull Clerk into bundles when the
  // env var is absent. Mirrors the gateway/Resend pattern.
  const { auth } = await import("@clerk/nextjs/server");
  const session = await auth();
  if (!session.userId) {
    return { ok: false, code: "unauthenticated" };
  }
  return { ok: true, userId: session.userId };
}

/**
 * JIT user provisioning — covers the brief window between Clerk's sign-up
 * completion and its webhook firing `user.created`. Idempotent: the row is
 * upserted with a placeholder email (the webhook backfills the real one).
 *
 * The placeholder email shape mirrors the spec — the Clerk webhook sends the
 * primary email address shortly after; until then, dispatch to a placeholder
 * address would `failed` at the Resend layer, which is acceptable for the
 * <2s gap.
 */
export async function ensureUserExists(
  db: AppDb,
  userId: string,
): Promise<void> {
  await db.execute(sql`
    INSERT INTO "users" ("id", "email")
    VALUES (${userId}, ${`${userId}@pending.invalid`})
    ON CONFLICT ("id") DO NOTHING
  `);
}
