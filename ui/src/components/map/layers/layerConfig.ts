// Static color scales, radius accessors, opacity, stroke widths for map layers.
// Pure data — no React.

export const BASEMAP_DARK = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
export const BASEMAP_LIGHT = "https://basemaps.cartocdn.com/gl/positron-nolabels-gl-style/style.json";
export const BASEMAP_SATELLITE = {
  version: 8 as const,
  sources: {
    "esri-satellite": {
      type: "raster" as const,
      tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
      tileSize: 256,
      attribution: "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
    }
  },
  layers: [{ id: "esri-satellite", type: "raster" as const, source: "esri-satellite" }]
};

export const FORECAST_FILL: [number, number, number, number] = [255, 165, 0, 40];
export const FORECAST_STROKE: [number, number, number, number] = [255, 165, 0, 200];

export const HIGH_CONFIDENCE_THRESHOLD = 0.6;

export const MIN_SELECTION_ZOOM = 6;
export const MAX_SELECTION_ZOOM = 14;
export const SELECTION_TARGET_OCCUPANCY = 0.3;

export const SELECTED_FRONT_COLOR: [number, number, number, number] = [59, 130, 246, 255];
export const SELECTED_EVENT_FILL: [number, number, number, number] = [59, 130, 246, 88];
export const SELECTED_EVENT_STROKE: [number, number, number, number] = [96, 165, 250, 255];

// Centroid scatterplot config
export const CENTROID_RADIUS_PX = 8;
export const CENTROID_RADIUS_MIN_PX = 4;
export const CENTROID_RADIUS_MAX_PX = 16;

// Event polygon stroke widths
export const EVENT_LINE_WIDTH = 3;
export const EVENT_LINE_MIN_PX = 1;
export const EVENT_LINE_MAX_PX = 4;

export const SELECTED_EVENT_LINE_WIDTH = 5;
export const SELECTED_EVENT_LINE_MIN_PX = 2;
export const SELECTED_EVENT_LINE_MAX_PX = 8;

// Front line widths
export const FRONT_LINE_MIN_PX = 1;
export const FRONT_LINE_MAX_PX = 6;

export const SELECTED_FRONT_LINE_MIN_PX = 2;
export const SELECTED_FRONT_LINE_MAX_PX = 10;
export const SELECTED_FRONT_LINE_WIDTH_BOOST = 2;
