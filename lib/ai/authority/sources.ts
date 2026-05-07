/**
 * Stage 8 — public authority-perimeter source catalog.
 *
 * Each source publishes recent fire perimeters as a public, key-free
 * GeoJSON FeatureCollection. The orchestrator (`lib/ai/generate.ts`)
 * pre-fetches the most recent feature near the detection and folds it into
 * the brief context. Path A from brief 22; tool-calling (Path B) is a v1.1
 * follow-up.
 *
 * Confirmed working endpoints (curl, 2026-05-07):
 *   - NIFC WFIGS Interagency Perimeters Current
 *     `https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson`
 *     Sample feature props: poly_PolygonDateTime (epoch ms),
 *     attr_FireDiscoveryDateTime (epoch ms), attr_IncidentName.
 *   - CWFIS m3_polygons_current (Natural Resources Canada)
 *     `https://cwfis.cfs.nrcan.gc.ca/geoserver/public/ows?service=WFS&version=2.0.0&request=GetFeature&typeNames=public:m3_polygons_current&outputFormat=application/json&srsName=EPSG:4326`
 *     Sample feature props: lastdate (ISO 8601 string), firstdate, area.
 *
 * NOT shipped in Stage 8 (filed as Vanyo blocker):
 *   - ICNF (Portugal): no key-free public GeoJSON perimeter endpoint found.
 *     fogos.pt publishes incident POINTS only, not polygons. ICNF SGIF is
 *     map-tile only. Per brief instructions: surfaced as a blocker, not
 *     fabricated.
 */

export type AuthoritySourceId = "nifc" | "cwfis";

export type AuthoritySource = {
  id: AuthoritySourceId;
  /** User-visible label persisted into `brief.context.authority_perimeter.source`. */
  name: string;
  /** Public GeoJSON URL. Confirmed key-free. */
  url: string;
  /**
   * Predicate against `region_bucket` (e.g. "5x5:W125_N40"). True if this
   * source's coverage area includes that 5°×5° tile. Cheaper and more
   * accurate than a hardcoded prefix list.
   */
  coversBucket: (bucket: string) => boolean;
  /**
   * Extract the publication timestamp from a feature's properties. Returns
   * an ISO 8601 string or null if the feature has no usable timestamp.
   */
  extractPostedTs: (props: Record<string, unknown>) => string | null;
  /** Source-stable id for the feature (debug only — never sent to the LLM). */
  extractFeatureId: (feature: { id?: unknown; properties?: Record<string, unknown> }) => string | undefined;
};

const NIFC: AuthoritySource = {
  id: "nifc",
  name: "NIFC WFIGS",
  url: "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/WFIGS_Interagency_Perimeters_Current/FeatureServer/0/query?where=1%3D1&outFields=*&f=geojson",
  coversBucket: (bucket) => {
    // CONUS + Alaska + Hawaii. Bucket format: 5x5:<E|W><lonAbs>_<N|S><latAbs>
    // CONUS roughly lon -125..-65, lat 24..50. Alaska lon -170..-130, lat 52..72.
    // Hawaii lon -160..-154, lat 18..23.
    const parsed = parseBucket(bucket);
    if (!parsed) return false;
    const { lon, lat } = parsed;
    if (lat >= 24 && lat <= 50 && lon >= -125 && lon <= -65) return true;
    if (lat >= 52 && lat <= 72 && lon >= -170 && lon <= -130) return true;
    if (lat >= 18 && lat <= 23 && lon >= -160 && lon <= -154) return true;
    return false;
  },
  extractPostedTs: (props) => {
    // Prefer the polygon-as-of timestamp, fall back to discovery time.
    const candidates = [
      props.poly_PolygonDateTime,
      props.poly_DateCurrent,
      props.attr_FireDiscoveryDateTime,
    ];
    for (const v of candidates) {
      if (typeof v === "number" && Number.isFinite(v)) {
        const d = new Date(v);
        if (Number.isFinite(d.getTime())) return d.toISOString();
      }
      if (typeof v === "string" && v) {
        const d = new Date(v);
        if (Number.isFinite(d.getTime())) return d.toISOString();
      }
    }
    return null;
  },
  extractFeatureId: (feature) => {
    if (typeof feature.id === "string" || typeof feature.id === "number") {
      return String(feature.id);
    }
    const irwin = feature.properties?.attr_IrwinID;
    if (typeof irwin === "string") return irwin;
    return undefined;
  },
};

const CWFIS: AuthoritySource = {
  id: "cwfis",
  name: "CWFIS",
  url: "https://cwfis.cfs.nrcan.gc.ca/geoserver/public/ows?service=WFS&version=2.0.0&request=GetFeature&typeNames=public:m3_polygons_current&outputFormat=application/json&srsName=EPSG:4326",
  coversBucket: (bucket) => {
    // Canada roughly lon -141..-52, lat 41..83.
    const parsed = parseBucket(bucket);
    if (!parsed) return false;
    const { lon, lat } = parsed;
    return lat >= 41 && lat <= 83 && lon >= -141 && lon <= -52;
  },
  extractPostedTs: (props) => {
    const v = props.lastdate ?? props.firstdate;
    if (typeof v === "string" && v) {
      const d = new Date(v);
      if (Number.isFinite(d.getTime())) return d.toISOString();
    }
    return null;
  },
  extractFeatureId: (feature) => {
    if (typeof feature.id === "string" || typeof feature.id === "number") {
      return String(feature.id);
    }
    return undefined;
  },
};

export const AUTHORITY_SOURCES: readonly AuthoritySource[] = [NIFC, CWFIS];

export function selectSourceForBucket(bucket: string): AuthoritySource | null {
  for (const s of AUTHORITY_SOURCES) {
    if (s.coversBucket(bucket)) return s;
  }
  return null;
}

/** Parse the SW corner of a 5°×5° bucket key. */
function parseBucket(bucket: string): { lon: number; lat: number } | null {
  const m = /^5x5:([EW])(\d{3})_([NS])(\d{2})$/.exec(bucket);
  if (!m) return null;
  const [, lonHemi, lonAbs, latHemi, latAbs] = m;
  const lon = (lonHemi === "W" ? -1 : 1) * Number(lonAbs);
  const lat = (latHemi === "S" ? -1 : 1) * Number(latAbs);
  if (!Number.isFinite(lon) || !Number.isFinite(lat)) return null;
  return { lon, lat };
}
