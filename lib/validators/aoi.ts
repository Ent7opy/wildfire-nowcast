/**
 * Zod schemas for AOI request / response shapes. Co-located with the routes
 * via re-export from the route files; defined here for reuse by tests.
 */
import { z } from "zod";
import { polygonalGeomSchema } from "./geojson";

const ianaTz = z
  .string()
  .min(1)
  .max(64)
  .regex(/^[A-Za-z][A-Za-z0-9_+\-/]*$/, "expected an IANA timezone string");

const hour = z.number().int().min(0).max(23);

export const quietHoursSchema = z
  .object({
    tz: ianaTz,
    startHour: hour,
    endHour: hour,
  })
  .nullable();

export const channelSchema = z.discriminatedUnion("type", [
  z.object({ type: z.literal("email"), target: z.string().email() }),
  z.object({ type: z.literal("webhook"), target: z.string().url() }),
]);

export const aoiCreateSchema = z.object({
  name: z.string().min(1).max(120),
  geometry: polygonalGeomSchema,
});
export type AoiCreate = z.infer<typeof aoiCreateSchema>;

export const aoiUpdateSchema = z
  .object({
    name: z.string().min(1).max(120).optional(),
    geometry: polygonalGeomSchema.optional(),
  })
  .refine(
    (v) => v.name !== undefined || v.geometry !== undefined,
    { message: "PATCH body must include at least one of name | geometry" },
  );
export type AoiUpdate = z.infer<typeof aoiUpdateSchema>;

export const rulesUpsertSchema = z.object({
  distanceBufferKm: z.number().positive().max(500).default(25),
  minConfidence: z.enum(["low", "nominal", "high"]).default("nominal"),
  minFrpMw: z.number().nonnegative().max(1000).default(5),
  quietHours: quietHoursSchema.default(null),
  pausedUntil: z
    .union([z.string().datetime(), z.null()])
    .default(null),
  notifyChannels: z.array(channelSchema).default([]),
});
export type RulesUpsert = z.infer<typeof rulesUpsertSchema>;
