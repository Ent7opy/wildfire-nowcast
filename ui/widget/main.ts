/**
 * Wildfire Forecast Widget
 *
 * Embeddable standalone widget that renders a spread forecast on a minimal map.
 *
 * Usage via script tag:
 *   <script
 *     src="/dist-widget/widget.js"
 *     data-run-id="42"
 *     data-api-base="https://api.example.com"
 *     data-horizon-hours="24"
 *     data-center-lon="20.5"
 *     data-center-lat="42.0"
 *     data-zoom="7"
 *   ></script>
 *
 * Usage via JS API:
 *   WildfireWidget.init({
 *     runId: 42,
 *     apiBase: 'https://api.example.com',
 *     horizonHours: 24,
 *     container: 'my-map-div',   // element id or HTMLElement
 *   });
 */

import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

export interface WidgetConfig {
  /** Forecast run ID. */
  runId: number;
  /** API base URL (no trailing slash). Defaults to same origin. */
  apiBase?: string;
  /** Forecast horizon in hours. Default: 24. */
  horizonHours?: number;
  /** Initial map center longitude. Default: 0. */
  centerLon?: number;
  /** Initial map center latitude. Default: 30. */
  centerLat?: number;
  /** Initial zoom level. Default: 4. */
  zoom?: number;
  /**
   * Container to render the map into.
   * Can be an element id string, an HTMLElement, or omitted to auto-append
   * a new div to <body>.
   */
  container?: string | HTMLElement;
  /** Map height when the widget creates its own container. Default: "400px". */
  height?: string;
}

function resolveContainer(config: WidgetConfig): HTMLElement {
  if (config.container instanceof HTMLElement) {
    return config.container;
  }
  if (typeof config.container === "string") {
    const el = document.getElementById(config.container);
    if (!el) {
      throw new Error(
        `WildfireWidget: container element "#${config.container}" not found`
      );
    }
    return el;
  }
  const div = document.createElement("div");
  div.style.cssText = `width:100%;height:${config.height ?? "400px"};`;
  document.body.appendChild(div);
  return div;
}

/**
 * Initialise a forecast widget and return the MapLibre map instance.
 */
export function init(config: WidgetConfig): maplibregl.Map {
  const {
    runId,
    apiBase = "",
    horizonHours = 24,
    centerLon = 0,
    centerLat = 30,
    zoom = 4,
  } = config;

  const container = resolveContainer(config);

  const map = new maplibregl.Map({
    container,
    style: {
      version: 8,
      sources: {
        osm: {
          type: "raster",
          tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
          tileSize: 256,
          attribution: "© OpenStreetMap contributors",
        },
      },
      layers: [{ id: "osm", type: "raster", source: "osm" }],
    },
    center: [centerLon, centerLat],
    zoom,
    attributionControl: true,
  });

  map.addControl(new maplibregl.NavigationControl(), "top-right");

  map.on("load", () => {
    const tileUrl =
      `${apiBase}/forecast/${runId}/tiles/{z}/{x}/{y}.png` +
      `?horizon_hours=${horizonHours}`;

    map.addSource("wildfire-forecast", {
      type: "raster",
      tiles: [tileUrl],
      tileSize: 256,
      attribution: "Wildfire Nowcast",
    });

    map.addLayer({
      id: "wildfire-forecast-layer",
      type: "raster",
      source: "wildfire-forecast",
      paint: { "raster-opacity": 0.85 },
    });
  });

  return map;
}

// ---------------------------------------------------------------------------
// Auto-initialise from script tag data attributes
// ---------------------------------------------------------------------------

let _autoInitDone = false;

function autoInit(): void {
  if (_autoInitDone) return;
  _autoInitDone = true;

  const scripts = document.querySelectorAll<HTMLScriptElement>(
    "script[data-run-id]"
  );

  scripts.forEach((script) => {
    const runId = parseInt(script.dataset.runId ?? "", 10);
    if (Number.isNaN(runId)) {
      console.warn("WildfireWidget: data-run-id is missing or not a number");
      return;
    }

    const horizonHours = script.dataset.horizonHours
      ? parseInt(script.dataset.horizonHours, 10)
      : 24;
    const centerLon = script.dataset.centerLon
      ? parseFloat(script.dataset.centerLon)
      : 0;
    const centerLat = script.dataset.centerLat
      ? parseFloat(script.dataset.centerLat)
      : 30;
    const zoom = script.dataset.zoom ? parseInt(script.dataset.zoom, 10) : 4;

    init({
      runId,
      apiBase: script.dataset.apiBase ?? "",
      horizonHours,
      centerLon,
      centerLat,
      zoom,
    });
  });
}

// Expose global API for programmatic use
(window as unknown as Record<string, unknown>).WildfireWidget = { init };

// Kick off auto-init once the DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", autoInit);
} else {
  autoInit();
}
