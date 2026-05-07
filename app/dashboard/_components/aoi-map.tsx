/**
 * Stage 7 — MapLibre map for AOI display + (optional) polygon draw.
 *
 * Client-only. Loaded via `next/dynamic({ ssr: false })` from the AOI pages
 * so MapLibre's ~250 KB gzipped bundle is not in the initial dashboard JS.
 *
 * Tile source: MapLibre demo style (`https://demotiles.maplibre.org/style.json`).
 * Free, no API key, attribution-required (MapLibre's default attribution
 * control covers it). When v1.1 swaps to a real basemap, change the style URL.
 *
 * Polygon-draw mode is in-house (~40 LOC) instead of importing
 * `@mapbox/mapbox-gl-draw` (which needs a maplibre adapter shim) or
 * `terra-draw` (heavier API surface than we need). Click adds vertices;
 * double-click closes the ring; the drawn polygon is handed to `onPolygon`
 * as a GeoJSON Polygon.
 *
 * Smoke-test friendly: jsdom has no WebGL, so MapLibre's `new Map(...)` will
 * throw. We catch that synchronously and render a "(map unavailable)" notice
 * so the smoke test asserts mount-without-crash, not pixel rendering.
 */
"use client";

import { useEffect, useRef, useState } from "react";
import type {
  GeoJSONMultiPolygon,
  GeoJSONPoint,
  GeoJSONPolygon,
} from "@/lib/geo/polygon";

type DetectionPoint = {
  lat: number;
  lon: number;
  frpMw: number | null;
  detectedAt: string;
  satellite: string;
};

export type AoiMapProps =
  | {
      mode: "view";
      polygon: GeoJSONPolygon | GeoJSONMultiPolygon;
      bbox: GeoJSONPolygon;
      centroid: GeoJSONPoint;
      detections?: DetectionPoint[];
    }
  | {
      mode: "draw";
      onPolygon: (polygon: GeoJSONPolygon) => void;
      initialCenter?: { lon: number; lat: number };
    };

const STYLE_URL = "https://demotiles.maplibre.org/style.json";

export function AoiMap(props: AoiMapProps): React.ReactElement {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let map: { remove: () => void } | null = null;
    let cancelled = false;

    (async () => {
      try {
        const maplibreMod = await import("maplibre-gl");
        await import("maplibre-gl/dist/maplibre-gl.css");
        const maplibre = maplibreMod.default ?? maplibreMod;
        if (cancelled || !containerRef.current) return;

        const center =
          props.mode === "view"
            ? (props.centroid.coordinates as [number, number])
            : props.initialCenter
              ? [props.initialCenter.lon, props.initialCenter.lat]
              : [0, 30];

        const m = new maplibre.Map({
          container: containerRef.current,
          style: STYLE_URL,
          center: center as [number, number],
          zoom: 4,
          attributionControl: { compact: true },
        });
        map = m;

        m.on("load", () => {
          if (props.mode === "view") {
            installViewLayers(maplibre, m, props);
          } else {
            installDrawHandlers(maplibre, m, props.onPolygon);
          }
        });
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      }
    })();

    return () => {
      cancelled = true;
      try {
        map?.remove();
      } catch {
        // ignore — jsdom path
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="w-full">
      <div
        ref={containerRef}
        data-testid="aoi-map-container"
        className="h-[420px] w-full rounded border border-[color:var(--muted)]"
        style={{ minHeight: 320 }}
      />
      {error ? (
        <p className="mt-2 text-xs text-[color:var(--muted)]">
          Map unavailable: {error}
        </p>
      ) : null}
    </div>
  );
}

/* eslint-disable @typescript-eslint/no-explicit-any */
function installViewLayers(
  maplibre: any,
  m: any,
  props: Extract<AoiMapProps, { mode: "view" }>,
): void {
  const polygonGeoJSON: GeoJSONPolygon | GeoJSONMultiPolygon = props.polygon;

  m.addSource("aoi", {
    type: "geojson",
    data: { type: "Feature", geometry: polygonGeoJSON, properties: {} },
  });
  m.addLayer({
    id: "aoi-fill",
    type: "fill",
    source: "aoi",
    paint: { "fill-color": "#1f5fb8", "fill-opacity": 0.15 },
  });
  m.addLayer({
    id: "aoi-line",
    type: "line",
    source: "aoi",
    paint: { "line-color": "#1f5fb8", "line-width": 2 },
  });

  const detections = props.detections ?? [];
  if (detections.length > 0) {
    m.addSource("detections", {
      type: "geojson",
      data: {
        type: "FeatureCollection",
        features: detections.map((d) => ({
          type: "Feature" as const,
          geometry: {
            type: "Point" as const,
            coordinates: [d.lon, d.lat],
          },
          properties: {
            frp: d.frpMw ?? 0,
            detectedAt: d.detectedAt,
            satellite: d.satellite,
          },
        })),
      },
    });
    m.addLayer({
      id: "detections-circle",
      type: "circle",
      source: "detections",
      paint: {
        "circle-radius": [
          "interpolate",
          ["linear"],
          ["coalesce", ["to-number", ["get", "frp"]], 0],
          0,
          3,
          50,
          10,
        ],
        "circle-color": "#d24343",
        "circle-opacity": 0.65,
        "circle-stroke-color": "#7a1d1d",
        "circle-stroke-width": 1,
      },
    });
  }

  // Fit to AOI bbox.
  const ring = props.bbox.coordinates[0];
  const lons = ring.map((p) => p[0]);
  const lats = ring.map((p) => p[1]);
  const bounds = new maplibre.LngLatBounds(
    [Math.min(...lons), Math.min(...lats)],
    [Math.max(...lons), Math.max(...lats)],
  );
  m.fitBounds(bounds, { padding: 32, maxZoom: 12 });
}

function installDrawHandlers(
  maplibre: any,
  m: any,
  onPolygon: (polygon: GeoJSONPolygon) => void,
): void {
  const points: Array<[number, number]> = [];

  m.addSource("draw-buffer", {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] },
  });
  m.addLayer({
    id: "draw-line",
    type: "line",
    source: "draw-buffer",
    paint: { "line-color": "#1f5fb8", "line-width": 2 },
  });
  m.addLayer({
    id: "draw-fill",
    type: "fill",
    source: "draw-buffer",
    paint: { "fill-color": "#1f5fb8", "fill-opacity": 0.15 },
  });

  const refresh = (closed: boolean) => {
    const coords = closed && points.length >= 3
      ? [...points, points[0]]
      : points;
    if (coords.length < 2) {
      m.getSource("draw-buffer").setData({
        type: "FeatureCollection",
        features: [],
      });
      return;
    }
    const geom = closed && points.length >= 3
      ? { type: "Polygon" as const, coordinates: [coords] }
      : { type: "LineString" as const, coordinates: coords };
    m.getSource("draw-buffer").setData({
      type: "Feature",
      geometry: geom,
      properties: {},
    });
  };

  m.on("click", (e: { lngLat: { lng: number; lat: number } }) => {
    points.push([e.lngLat.lng, e.lngLat.lat]);
    refresh(false);
  });

  m.on("dblclick", (e: { preventDefault?: () => void }) => {
    if (e.preventDefault) e.preventDefault();
    if (points.length < 3) return;
    const ring: [number, number][] = [...points, points[0]];
    refresh(true);
    onPolygon({ type: "Polygon", coordinates: [ring] });
  });

  // Tell MapLibre not to zoom on dblclick — the user wants to close the ring.
  if (m.doubleClickZoom?.disable) m.doubleClickZoom.disable();

  // Hint shown at bottom-left
  const ctrl = new maplibre.AttributionControl({
    customAttribution: "Click to add vertices · double-click to finish",
  });
  m.addControl(ctrl, "bottom-left");
}
/* eslint-enable @typescript-eslint/no-explicit-any */
