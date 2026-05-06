/**
 * Shared request-handling helpers for the AOI routes. Owns the parse → auth →
 * repo → map-error → response funnel so the route files stay short.
 *
 * Stage 5: `withDb` is now async and authenticated. It resolves the calling
 * Clerk user via `requireUserId()`, JIT-provisions a `users` row (covers the
 * webhook-lag race), and forwards `{ db, userId }` to the route handler.
 *
 * Build-without-blocking: when `CLERK_SECRET_KEY` is unset the route returns
 * a typed 503 `service_unavailable`; the rest of the app continues to build
 * and start.
 */
import { NextResponse, type NextRequest } from "next/server";
import { ZodError, type ZodType } from "zod";
import {
  apiError,
  dbUnavailableResponse,
  zodErrorResponse,
  type ApiErrorBody,
} from "./errors";
import { tryGetDb, type AppDb } from "@/lib/db/client";
import { ensureUserExists, requireUserId } from "@/lib/auth/context";
import {
  AoiAreaTooLargeError,
  AoiNameConflictError,
  AoiNotFoundError,
} from "@/lib/db/aoi-repository";

export type WithDbContext = {
  db: AppDb;
  userId: string;
};

export async function withDb<T>(
  handler: (ctx: WithDbContext) => Promise<NextResponse<T> | NextResponse<ApiErrorBody>>,
): Promise<NextResponse<T> | NextResponse<ApiErrorBody>> {
  const db = tryGetDb();
  if (!db) return dbUnavailableResponse();
  const auth = await requireUserId();
  if (!auth.ok) {
    if (auth.code === "config_missing") {
      return apiError(
        "service_unavailable",
        "Auth not configured; CLERK_SECRET_KEY is unset",
      );
    }
    return apiError("unauthenticated", "Sign in required");
  }
  try {
    await ensureUserExists(db, auth.userId);
    return await handler({ db, userId: auth.userId });
  } catch (err) {
    return mapDomainError(err);
  }
}

export async function parseJson<T>(
  req: NextRequest,
  schema: ZodType<T>,
): Promise<{ ok: true; value: T } | { ok: false; response: NextResponse<ApiErrorBody> }> {
  let raw: unknown;
  try {
    raw = await req.json();
  } catch {
    return {
      ok: false,
      response: apiError("validation_failed", "Request body must be valid JSON"),
    };
  }
  const parsed = schema.safeParse(raw);
  if (!parsed.success) {
    return { ok: false, response: zodErrorResponse(parsed.error) };
  }
  return { ok: true, value: parsed.data };
}

function mapDomainError(err: unknown): NextResponse<ApiErrorBody> {
  if (err instanceof ZodError) return zodErrorResponse(err);
  if (err instanceof AoiNotFoundError) {
    return apiError("not_found", err.message);
  }
  if (err instanceof AoiNameConflictError) {
    return apiError("conflict", err.message);
  }
  if (err instanceof AoiAreaTooLargeError) {
    return apiError("validation_failed", err.message, {
      areaHa: err.areaHa,
    });
  }
  console.error("[api] unhandled error:", err);
  return apiError("internal_error", "Unexpected server error");
}
