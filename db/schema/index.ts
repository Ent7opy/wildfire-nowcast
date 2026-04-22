/**
 * Stage 1 schema — A' (Fire Stewardship Agent).
 *
 * Source of truth: `docs/pivot-architecture.md` §3 "The collapsed data model".
 * Spec: `docs/SPEC-A-prime-v1.md` §Data model.
 *
 * Stage 1 (this file): `users`, `aois`, `aoi_rules` — enough for AOI CRUD.
 *
 * DEFERRED to later stages (intentionally omitted here):
 *   - `firms_detections`     → Stage 2 (FIRMS poll + match)
 *   - `aoi_events`           → Stage 2 (detection ↔ AOI matches)
 *   - `aoi_briefs`           → Stage 3 (LLM situation briefs)
 *   - `notifications_log`    → Stage 4 (Resend + webhook delivery)
 *   - `industrial_mask_static` → Stage 2 (seeded GeoJSON)
 *   - `job_runs`             → Stage 2 (cron observability)
 *
 * Single-user stub: until Clerk lands in Stage 5, every API call is attributed
 * to STUB_USER_ID = "stub-user-1". The migration seeds that single row.
 */
import {
  customType,
  jsonb,
  pgTable,
  real,
  text,
  timestamp,
  uniqueIndex,
  uuid,
} from "drizzle-orm/pg-core";
import { sql } from "drizzle-orm";
import { geometry } from "./postgis";

/**
 * `bytea` for encrypted Gemini BYO key (Stage 5+).
 * Modeled now to keep migrations stable; never returned to clients.
 */
const bytea = customType<{ data: Buffer; driverData: Buffer }>({
  dataType: () => "bytea",
});

export const STUB_USER_ID = "stub-user-1";

export const users = pgTable("users", {
  id: text("id").primaryKey(), // Clerk user_id; for now: STUB_USER_ID
  email: text("email").notNull(),
  displayName: text("display_name"),
  geminiApiKeyEnc: bytea("gemini_api_key_enc"),
  createdAt: timestamp("created_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
  deletedAt: timestamp("deleted_at", { withTimezone: true }),
});

export const aois = pgTable(
  "aois",
  {
    id: uuid("id")
      .primaryKey()
      .default(sql`gen_random_uuid()`),
    userId: text("user_id")
      .notNull()
      .references(() => users.id, { onDelete: "cascade" }),
    name: text("name").notNull(),
    polygon: geometry("polygon", { srid: 4326, subtype: "MultiPolygon" }).notNull(),
    bbox: geometry("bbox", { srid: 4326, subtype: "Polygon" }).notNull(),
    centroid: geometry("centroid", { srid: 4326, subtype: "Point" }).notNull(),
    /**
     * Coarse 5°×5° tile key, derived from the centroid by floor() at create
     * time. Used by Stage 2's cron to coalesce FIRMS calls. Format: e.g.
     * "5x5:W015_N045" — see `lib/geo/region-bucket.ts`.
     */
    regionBucket: text("region_bucket").notNull(),
    /**
     * Denormalized polygon area in hectares, computed at create time via
     * PostGIS `ST_Area(polygon::geography) / 10000`. Cached for the v1 spec
     * area cap (≤100,000 ha) and listing UI.
     */
    areaHa: real("area_ha").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true })
      .notNull()
      .defaultNow(),
    archivedAt: timestamp("archived_at", { withTimezone: true }),
  },
  (table) => [
    // Spatial + lookup indexes — created in the SQL migration via raw `CREATE
    // INDEX ... USING GIST`. Drizzle's index DSL doesn't natively express
    // GIST yet, so we declare them in the SQL migration alongside this file.
    uniqueIndex("aois_user_name_active_uniq")
      .on(table.userId, table.name)
      .where(sql`archived_at IS NULL`),
  ],
);

export const aoiRules = pgTable("aoi_rules", {
  aoiId: uuid("aoi_id")
    .primaryKey()
    .references(() => aois.id, { onDelete: "cascade" }),
  /** Distance buffer in km used by Stage 2 matcher. Spec default: 25. */
  distanceBufferKm: real("distance_buffer_km").notNull().default(25),
  /** "low" | "nominal" | "high" — passed through to FIRMS confidence gate. */
  minConfidence: text("min_confidence").notNull().default("nominal"),
  /** Minimum FRP (MW) for the LLM gate (Stage 3). */
  minFrpMw: real("min_frp_mw").notNull().default(5),
  /**
   * Quiet hours, modeled as a JSONB blob to keep the DB layer flexible while
   * the v1 UX settles. Validated by Zod at the API boundary.
   *   { tz: "America/Los_Angeles", startHour: 22, endHour: 7 }
   */
  quietHours: jsonb("quiet_hours").$type<{
    tz: string;
    startHour: number;
    endHour: number;
  } | null>(),
  /**
   * Pause AOI delivery (history still accumulates).
   * Reserved for the v1 snooze/pause flow (US-5).
   */
  pausedUntil: timestamp("paused_until", { withTimezone: true }),
  /**
   * Notification channels.
   *   [{ type: "email", target: "..." }, { type: "webhook", target: "https://..." }]
   */
  notifyChannels: jsonb("notify_channels")
    .$type<
      Array<
        | { type: "email"; target: string }
        | { type: "webhook"; target: string }
      >
    >()
    .notNull()
    .default(sql`'[]'::jsonb`),
  updatedAt: timestamp("updated_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
});

