/**
 * PostGIS custom column types for Drizzle ORM.
 *
 * Drizzle does not ship native PostGIS types; we model them via `customType`.
 * Storage is the canonical PostGIS `geometry` type with explicit subtype +
 * SRID 4326. We exchange GeoJSON in TypeScript and let PostGIS convert via
 * `ST_GeomFromGeoJSON` / `ST_AsGeoJSON` in raw SQL at the call sites
 * (see `db/repositories/aoi.ts`).
 *
 * The driver value here is `string` because the column is read/written as
 * SQL expressions (e.g. `ST_GeomFromGeoJSON(...)`), not as a literal value.
 * Always interpolate via `sql` template tags — never via raw concatenation.
 */
import { customType } from "drizzle-orm/pg-core";

type GeometryConfig = {
  srid: number;
  subtype: "Polygon" | "MultiPolygon" | "Point";
};

export const geometry = customType<{
  data: string; // app passes WKT or relies on sql`ST_GeomFromGeoJSON(...)` writes
  driverData: string;
  config: GeometryConfig;
}>({
  dataType(config) {
    if (!config) {
      throw new Error("geometry() requires { srid, subtype }");
    }
    return `geometry(${config.subtype}, ${config.srid})`;
  },
});
