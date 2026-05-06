/**
 * Typed API error envelope: `{ error: { code, message, details? } }`.
 *
 * Stage 1 keeps the codes intentionally small. Later stages may extend them
 * (e.g. `gate_failed` for Stage 3 LLM gate). All routes funnel through
 * `apiError` so the envelope stays consistent.
 */
import { NextResponse } from "next/server";
import { ZodError } from "zod";

export type ApiErrorCode =
  | "validation_failed"
  | "unauthenticated"
  | "not_found"
  | "conflict"
  | "service_unavailable"
  | "internal_error";

export type ApiErrorBody = {
  error: {
    code: ApiErrorCode;
    message: string;
    details?: unknown;
  };
};

const STATUS: Record<ApiErrorCode, number> = {
  validation_failed: 400,
  unauthenticated: 401,
  not_found: 404,
  conflict: 409,
  service_unavailable: 503,
  internal_error: 500,
};

export function apiError(
  code: ApiErrorCode,
  message: string,
  details?: unknown,
): NextResponse<ApiErrorBody> {
  const body: ApiErrorBody = { error: { code, message } };
  if (details !== undefined) body.error.details = details;
  return NextResponse.json(body, { status: STATUS[code] });
}

export function zodErrorResponse(err: ZodError): NextResponse<ApiErrorBody> {
  return apiError("validation_failed", "Request payload failed validation", {
    issues: err.issues.map((i) => ({
      path: i.path,
      message: i.message,
      code: i.code,
    })),
  });
}

export function dbUnavailableResponse(): NextResponse<ApiErrorBody> {
  return apiError(
    "service_unavailable",
    "DATABASE_URL not configured; this is expected during pre-Neon-setup development",
  );
}
