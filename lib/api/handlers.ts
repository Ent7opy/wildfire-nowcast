/**
 * Shared request-handling helpers for the AOI routes. Owns the parse → repo →
 * map-error → response funnel so the route files stay short and uniform.
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
import { currentUserId } from "./current-user";
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
  try {
    return await handler({ db, userId: currentUserId() });
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

export function mapDomainError(err: unknown): NextResponse<ApiErrorBody> {
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
  // Unknown — log via console.error so Vercel surfaces it; respond with a
  // generic message (no leaking stack traces).
  console.error("[api] unhandled error:", err);
  return apiError("internal_error", "Unexpected server error");
}
