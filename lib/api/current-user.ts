/**
 * Current-user resolver — single-user stub for Stages 1–4.
 *
 * Stage 5 swaps this for `auth()` from `@clerk/nextjs/server`. Until then,
 * every request is attributed to STUB_USER_ID. Centralising here means the
 * Stage 5 swap touches one file.
 */
import { STUB_USER_ID } from "@/db/schema";

export function currentUserId(): string {
  return STUB_USER_ID;
}
