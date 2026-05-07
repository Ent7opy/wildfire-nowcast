/**
 * /api/me — return the calling user's profile.
 *
 * Spec: docs/SPEC-A-prime-v1.md §API surface (US-1).
 */
import { NextResponse } from "next/server";
import { sql } from "drizzle-orm";
import { withDb } from "@/lib/api/handlers";
import { apiError } from "@/lib/api/errors";
import { decodeRows } from "@/lib/db/decode-rows";

export const runtime = "nodejs";

export async function GET(): Promise<NextResponse> {
  return withDb(async ({ db, userId }) => {
    const result = await db.execute(sql`
      SELECT "id", "email", "gemini_api_key_enc"
      FROM "users"
      WHERE "id" = ${userId}
      LIMIT 1
    `);
    const rows = decodeRows<{
      id: string;
      email: string;
      gemini_api_key_enc: unknown;
    }>(result);
    const row = rows[0];
    if (!row) return apiError("not_found", "User row missing");
    return NextResponse.json({
      id: row.id,
      email: row.email,
      hasByoKey: row.gemini_api_key_enc != null,
    });
  });
}
