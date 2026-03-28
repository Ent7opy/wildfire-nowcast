import type {
  ReviewQueueResponse,
  ResolveReviewResponse
} from "../types/api";
import { getJson, postJson } from "./http";

export async function getDenoiserReviewQueue(): Promise<ReviewQueueResponse> {
  return getJson<ReviewQueueResponse>("/internal/denoiser/review-queue", { limit: 200, status: "open" });
}

export async function resolveDenoiserReviewItem(args: {
  eventId: string;
  resolvedBy: string;
  resolvedNotes?: string;
}): Promise<ResolveReviewResponse> {
  return postJson<ResolveReviewResponse>(
    `/internal/denoiser/review-queue/${args.eventId}/resolve`,
    {
      resolved_by: args.resolvedBy,
      resolved_notes: args.resolvedNotes ?? null
    }
  );
}
