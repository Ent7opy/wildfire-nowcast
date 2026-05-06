/**
 * Vercel AI Gateway client for Stage 3 brief generation.
 *
 * Single function: `generateBriefViaGateway` calls `generateObject` against
 * Gemini 2.5 Flash-Lite through the AI Gateway, with the v1 brief schema as
 * the structured-output contract.
 *
 * Build-without-blocking: `AI_GATEWAY_API_KEY` is read lazily. If unset we
 * return `{ ok: false, code: "config_missing" }` rather than throwing — the
 * orchestrator surfaces that as a `briefSkipReason`.
 *
 * The model id is fixed to `google/gemini-2.5-flash-lite` per ADR 0006 / SPEC.
 * Override via the `model` arg for tests / future model bumps.
 */
import { generateObject, NoObjectGeneratedError } from "ai";
import { createGateway } from "@ai-sdk/gateway";
import type { LanguageModel } from "ai";
import { BriefSchema, type Brief } from "./schema";

export const DEFAULT_MODEL_ID = "google/gemini-2.5-flash-lite";

export type GatewayErrCode =
  | "config_missing"
  | "schema_invalid"
  | "upstream_error"
  | "no_object_generated";

export type GatewayResult =
  | { ok: true; brief: Brief; modelId: string }
  | { ok: false; code: GatewayErrCode; message: string };

export type GenerateBriefArgs = {
  systemPrompt: string;
  userPrompt: string;
  /** Override the model id (default: Gemini 2.5 Flash-Lite). */
  modelId?: string;
  /** Inject a model for tests; bypasses the gateway entirely. */
  modelOverride?: LanguageModel;
  /** Override env-var read for tests. */
  apiKey?: string;
  /** Hard timeout for the gateway call, ms. Default: 30s. */
  timeoutMs?: number;
};

export async function generateBriefViaGateway(
  args: GenerateBriefArgs,
): Promise<GatewayResult> {
  const modelId = args.modelId ?? DEFAULT_MODEL_ID;

  let model: LanguageModel;
  if (args.modelOverride) {
    model = args.modelOverride;
  } else {
    const apiKey = args.apiKey ?? process.env.AI_GATEWAY_API_KEY;
    if (!apiKey) {
      return {
        ok: false,
        code: "config_missing",
        message:
          "AI_GATEWAY_API_KEY is not set (build-without-blocking pattern).",
      };
    }
    const gateway = createGateway({ apiKey });
    model = gateway(modelId);
  }

  try {
    const result = await generateObject({
      model,
      schema: BriefSchema,
      schemaName: "AoiBrief",
      schemaDescription:
        "L2 situation brief for a stewardship AOI; matches docs/SPEC-A-prime-v1.md §LLM brief format v1.",
      system: args.systemPrompt,
      prompt: args.userPrompt,
      abortSignal: args.timeoutMs
        ? AbortSignal.timeout(args.timeoutMs)
        : undefined,
    });
    // Defence in depth: re-validate even though `generateObject` already did.
    const reparsed = BriefSchema.safeParse(result.object);
    if (!reparsed.success) {
      return {
        ok: false,
        code: "schema_invalid",
        message: `Gateway returned schema-invalid object: ${reparsed.error.issues
          .map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`)
          .slice(0, 3)
          .join("; ")}`,
      };
    }
    return { ok: true, brief: reparsed.data, modelId };
  } catch (err) {
    if (NoObjectGeneratedError.isInstance(err)) {
      return {
        ok: false,
        code: "no_object_generated",
        message: err.message,
      };
    }
    return {
      ok: false,
      code: "upstream_error",
      message: err instanceof Error ? err.message : String(err),
    };
  }
}
