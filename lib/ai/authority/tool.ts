/**
 * Stage 8 — Path A wrapper around `fetchAuthorityPerimeter`.
 *
 * Despite the filename, this is NOT a Vercel AI SDK `tool()` call. AI SDK v6's
 * `generateObject` does not accept a `tools` parameter — tools are only on
 * `generateText` / `streamText`. Brief 22 §"Critical implementation question"
 * resolved this to **Path A**: the orchestrator pre-fetches before calling
 * the model, so the structured-output contract is preserved and there's only
 * one LLM round-trip per brief.
 *
 * The wrapper exists so that a v1.1 swap to a literal LLM tool-call is a
 * one-call change: the shape this returns is exactly what a `tool().execute`
 * would resolve to.
 */
import { fetchAuthorityPerimeter, type FetchPerimeterArgs } from "./fetch";

export type PerimeterToolResult = {
  source: string | null;
  posted_ts: string | null;
  contains_detection: boolean | null;
};

export async function runAuthorityPerimeterTool(
  args: FetchPerimeterArgs,
): Promise<PerimeterToolResult> {
  const r = await fetchAuthorityPerimeter(args);
  if (!r) return { source: null, posted_ts: null, contains_detection: null };
  return {
    source: r.source,
    posted_ts: r.postedTs,
    contains_detection: r.containsDetection,
  };
}
