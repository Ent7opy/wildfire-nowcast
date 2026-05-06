/**
 * A' (Fire Stewardship Agent) — Drizzle schema.
 *
 * Source of truth: `docs/pivot-architecture.md` §3 "The collapsed data model".
 * Spec: `docs/SPEC-A-prime-v1.md` §Data model.
 *
 * Stages landed in this file:
 *   - Stage 1: `users`, `aois`, `aoi_rules` — AOI CRUD.
 *   - Stage 2: `firms_detections`, `aoi_events`, `industrial_mask_static`,
 *              `job_runs` — FIRMS poll + AOI matcher + cron observability.
 *   - Stage 3: `aoi_briefs` + `aoi_events.last_brief_at` — LLM situation briefs.
 *
 * DEFERRED to later stages (intentionally omitted here):
 *   - `notifications_log`    → Stage 4 (Resend + webhook delivery)
 *
 * Single-user stub: until Clerk lands in Stage 5, every API call is attributed
 * to STUB_USER_ID = "stub-user-1". The migration seeds that single row.
 */
import {
  bigserial,
  boolean,
  customType,
  doublePrecision,
  integer,
  jsonb,
  numeric,
  pgTable,
  real,
  serial,
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

// ---------------------------------------------------------------------------
// Stage 2: FIRMS poll, matcher, cron observability
// ---------------------------------------------------------------------------

/**
 * Short-TTL (~14 days) cache of FIRMS pixels relevant to active AOIs.
 *
 * Source name uses NASA's enum: VIIRS_NOAA20_NRT, VIIRS_SNPP_NRT, MODIS_NRT.
 * Dedupe via the unique index on (source, acq_date, acq_time, lat, lon) —
 * geometry equality on a Point is brittle across drivers, so we use the
 * lat/lon doubles as the dedupe key and keep the GIST geometry for spatial
 * queries.
 */
export const firmsDetections = pgTable("firms_detections", {
  id: bigserial("id", { mode: "bigint" }).primaryKey(),
  source: text("source").notNull(),
  detectedAt: timestamp("detected_at", { withTimezone: true }).notNull(),
  geom: geometry("geom", { srid: 4326, subtype: "Point" }).notNull(),
  lat: doublePrecision("lat").notNull(),
  lon: doublePrecision("lon").notNull(),
  frpMw: real("frp_mw"),
  confidence: text("confidence"),
  daynight: text("daynight"),
  /** ISO date as TEXT, mirrors FIRMS CSV column verbatim for traceability. */
  acqDate: text("acq_date").notNull(),
  /** "HHMM" string from FIRMS CSV. */
  acqTime: text("acq_time").notNull(),
  brightTi4: real("bright_ti4"),
  brightTi5: real("bright_ti5"),
  scan: real("scan"),
  track: real("track"),
  version: text("version"),
  /**
   * Set at insert via STA mask lookup (industrial_mask_static). When true,
   * matcher skips this detection — it's a known industrial heat source.
   * NULL means "not yet evaluated"; the matcher treats NULL as false.
   */
  isIndustrialStatic: boolean("is_industrial_static"),
  /** 5°×5° bucket the detection was fetched in. */
  bucket: text("bucket").notNull(),
  insertedAt: timestamp("inserted_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
});

/**
 * Per-AOI matched events. One row per "event" — defined as a contiguous burst
 * of detections within a 24h window near the AOI. Re-poll within the window
 * UPDATEs the existing row (extends last_seen_at, bumps detection_count);
 * a new window or a moved cluster creates a new row.
 *
 * `dedupe_hash` is the de-duplication key: same hash => same event (UPSERT);
 * different hash => new event (Stage 3 will pick it up for brief generation).
 */
export const aoiEvents = pgTable("aoi_events", {
  id: uuid("id")
    .primaryKey()
    .default(sql`gen_random_uuid()`),
  aoiId: uuid("aoi_id")
    .notNull()
    .references(() => aois.id, { onDelete: "cascade" }),
  firstSeenAt: timestamp("first_seen_at", { withTimezone: true }).notNull(),
  lastSeenAt: timestamp("last_seen_at", { withTimezone: true }).notNull(),
  nearestDistanceKm: real("nearest_distance_km").notNull(),
  detectionCount: integer("detection_count").notNull().default(1),
  peakFrpMw: real("peak_frp_mw"),
  /** sha256 over (aoi_id, bucket, rounded centroid, source, 24h-window). */
  dedupeHash: text("dedupe_hash").notNull(),
  /** "new" | "open" | "closed". Stage 3's brief generator picks up "new". */
  status: text("status").notNull().default("new"),
  closedAt: timestamp("closed_at", { withTimezone: true }),
  /** Set when Stage 3 generates a brief for this event. Used by the gate. */
  lastBriefAt: timestamp("last_brief_at", { withTimezone: true }),
  createdAt: timestamp("created_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
});

/**
 * Stage 3 — generated situation briefs.
 *
 * One row per event that passed the gate. The Zod-validated JSON payload lives
 * in `payload`; `rendered_markdown` is the email/web body produced by the
 * deterministic Markdown renderer. `gate_reason` records which of the four
 * SPEC §Flow 6 conditions fired so we can audit gate pass-rate post-launch.
 */
export const aoiBriefs = pgTable("aoi_briefs", {
  id: uuid("id")
    .primaryKey()
    .default(sql`gen_random_uuid()`),
  aoiId: uuid("aoi_id")
    .notNull()
    .references(() => aois.id, { onDelete: "cascade" }),
  eventId: uuid("event_id")
    .notNull()
    .references(() => aoiEvents.id, { onDelete: "cascade" }),
  schemaVersion: integer("schema_version").notNull().default(1),
  /** Model id used for the generation, e.g. "google/gemini-2.5-flash-lite". */
  model: text("model").notNull(),
  /** Prompt template version pinned in `lib/ai/prompt.ts`, e.g. "v1". */
  promptVersion: text("prompt_version").notNull().default("v1"),
  /** Gate condition that triggered this brief: see lib/ai/gate.ts GateReason. */
  gateReason: text("gate_reason").notNull(),
  payload: jsonb("payload").notNull(),
  renderedMarkdown: text("rendered_markdown").notNull(),
  /** Estimated USD cost reported by the AI Gateway response, if available. */
  costUsdEst: numeric("cost_usd_est", { precision: 10, scale: 6 }),
  /** Wall-clock latency of the gateway call in milliseconds. */
  latencyMs: integer("latency_ms"),
  shareToken: text("share_token"),
  shareExpiresAt: timestamp("share_expires_at", { withTimezone: true }),
  createdAt: timestamp("created_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
});

/**
 * Static catalog of industrial / volcanic heat sources to suppress at insert
 * time. Seeded from `db/seeds/industrial-mask-stage2.json` during migration.
 * A future stage can ingest the full NASA STA mask layer.
 */
export const industrialMaskStatic = pgTable("industrial_mask_static", {
  id: serial("id").primaryKey(),
  kind: text("kind").notNull(), // gas_flare | industrial | volcanic | refinery
  name: text("name").notNull(),
  geom: geometry("geom", { srid: 4326, subtype: "Polygon" }).notNull(),
  sourceUrl: text("source_url"),
  loadedAt: timestamp("loaded_at", { withTimezone: true })
    .notNull()
    .defaultNow(),
});

/**
 * Cron observability — one row per `/api/aoi/poll` invocation, plus per-bucket
 * children when the poll fans out. Feeds the future "last N runs" admin UI.
 */
export const jobRuns = pgTable("job_runs", {
  id: bigserial("id", { mode: "bigint" }).primaryKey(),
  jobName: text("job_name").notNull(), // "firms-poll" for Stage 2
  bucket: text("bucket"), // null for the parent run, set for per-bucket children
  startedAt: timestamp("started_at", { withTimezone: true }).notNull(),
  finishedAt: timestamp("finished_at", { withTimezone: true }),
  status: text("status").notNull(), // ok | partial | error | running
  firmsRequestCount: integer("firms_request_count").notNull().default(0),
  detectionsInserted: integer("detections_inserted").notNull().default(0),
  eventsCreated: integer("events_created").notNull().default(0),
  /** Stage 3: count of `aoi_briefs` rows produced by this run. */
  briefsGenerated: integer("briefs_generated").notNull().default(0),
  error: text("error"),
});

export type FirmsDetectionRow = typeof firmsDetections.$inferSelect;
export type AoiEventRow = typeof aoiEvents.$inferSelect;
export type JobRunRow = typeof jobRuns.$inferSelect;
export type AoiBriefRow = typeof aoiBriefs.$inferSelect;
