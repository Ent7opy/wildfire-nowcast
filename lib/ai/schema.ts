/**
 * Stage 3 brief Zod schema — cloned verbatim from
 * `docs/SPEC-A-prime-v1.md` §LLM brief format.
 *
 * Single source of truth for:
 *   - Vercel AI SDK `generateObject({ schema: BriefSchema })` structured output
 *   - server-side re-validation after the LLM responds (defence in depth)
 *   - persisted `aoi_briefs.payload` JSON shape
 *   - the markdown renderer's input contract
 *
 * If the schema changes, bump SCHEMA_VERSION and write a migration. v1 stays
 * frozen at schema_version = 1.
 */
import { z } from "zod";

export const SCHEMA_VERSION = 1 as const;

export const BriefAoiSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1),
  area_ha: z.number().nonnegative(),
});

export const BriefKeyFactsSchema = z.object({
  nearest_detection_km: z.number().nonnegative(),
  bearing_from_aoi_deg: z.number().min(0).max(360).nullable(),
  wind_dir_deg: z.number().min(0).max(360).nullable(),
  wind_speed_kmh: z.number().nonnegative().nullable(),
  wind_toward_aoi: z.boolean().nullable(),
  detection_count_in_window: z.number().int().nonnegative(),
  max_frp_mw: z.number().nonnegative().nullable(),
  satellites: z.array(z.string()),
  window_hours: z.number().int().nonnegative(),
});

export const BriefAuthorityPerimeterSchema = z.object({
  source: z.string().nullable(),
  posted_ts: z.string().datetime({ offset: true }).nullable(),
  contains_detection: z.boolean().nullable(),
});

export const BriefPriorEventSchema = z.object({
  date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  description: z.string().min(1),
  outcome: z.string().nullable(),
});

export const BriefContextSchema = z.object({
  weather_note: z.string().nullable(),
  authority_perimeter: BriefAuthorityPerimeterSchema,
  prior_events: z.array(BriefPriorEventSchema),
});

export const BriefNextHintSchema = z.object({
  when: z.string().min(1),
  trigger: z.string().min(1),
});

export const BriefSchema = z.object({
  schema_version: z.literal(SCHEMA_VERSION),
  aoi: BriefAoiSchema,
  summary: z.string().min(1),
  key_facts: BriefKeyFactsSchema,
  context: BriefContextSchema,
  recommended_watch_items: z.array(z.string().min(1)),
  uncertainty: z.string().min(1),
  next_brief_hint: BriefNextHintSchema,
});

export type Brief = z.infer<typeof BriefSchema>;
