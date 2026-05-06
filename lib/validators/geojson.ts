/**
 * Zod schemas for GeoJSON shapes accepted by the AOI API.
 *
 * Aligned with the v1 spec (docs/SPEC-A-prime-v1.md US-1):
 *   - SRID 4326, lon/lat order
 *   - polygons up to 100,000 ha (enforced after parse, in the route handler)
 *   - Polygon or MultiPolygon accepted; normalised to MultiPolygon at write
 *
 * Coordinate-range validation is intentionally strict at the Zod layer:
 *   lon ∈ [-180, 180], lat ∈ [-90, 90]. PostGIS would reject violators
 *   anyway, but rejecting earlier produces cleaner error messages.
 */
import { z } from "zod";

const lon = z.number().gte(-180).lte(180);
const lat = z.number().gte(-90).lte(90);
const position = z.tuple([lon, lat]);

const linearRing = z
  .array(position)
  .min(4, "ring must have ≥4 positions (closing point)")
  .refine(
    (ring) => {
      const first = ring[0];
      const last = ring[ring.length - 1];
      return first[0] === last[0] && first[1] === last[1];
    },
    { message: "ring must be closed (first == last position)" },
  );

const polygonRings = z.array(linearRing).min(1);

export const polygonSchema = z.object({
  type: z.literal("Polygon"),
  coordinates: polygonRings,
});

export const multiPolygonSchema = z.object({
  type: z.literal("MultiPolygon"),
  coordinates: z.array(polygonRings).min(1),
});

export const polygonalGeomSchema = z.discriminatedUnion("type", [
  polygonSchema,
  multiPolygonSchema,
]);

export type PolygonalGeom = z.infer<typeof polygonalGeomSchema>;
