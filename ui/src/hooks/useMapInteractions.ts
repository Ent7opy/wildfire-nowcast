import type { PickingInfo } from "@deck.gl/core";
import { normalizePickedEvent } from "../utils/selection";
import { geometryBounds } from "../map/layerUtils";
import { selectionViewFromBounds } from "../utils/mapSelection";
import { useAppStore } from "../state/store";
import {
  MIN_SELECTION_ZOOM,
  MAX_SELECTION_ZOOM,
  SELECTION_TARGET_OCCUPANCY
} from "../components/map/layers/layerConfig";
import { geometryProvenanceLabel } from "../components/fire-details/types";

export function useMapInteractions() {
  const mapView = useAppStore((s) => s.mapView);
  const setMapView = useAppStore((s) => s.setMapView);
  const setSelectedEvent = useAppStore((s) => s.setSelectedEvent);
  const setLastClick = useAppStore((s) => s.setLastClick);
  const focusMapOnPoint = useAppStore((s) => s.focusMapOnPoint);

  const onClick = (info: PickingInfo): void => {
    const selected = normalizePickedEvent(info.object);
    if (!selected) {
      return;
    }

    const lat = Number(selected.lat);
    const lon = Number(selected.lon);
    setSelectedEvent(selected);
    setLastClick({ lat, lng: lon });
    const selectedBounds = geometryBounds(selected.geom_geojson);
    if (selectedBounds) {
      const next = selectionViewFromBounds(selectedBounds, {
        minZoom: MIN_SELECTION_ZOOM,
        maxZoom: MAX_SELECTION_ZOOM,
        targetOccupancy: SELECTION_TARGET_OCCUPANCY
      });
      setMapView({
        ...mapView,
        latitude: next.latitude,
        longitude: next.longitude,
        zoom: next.zoom,
        transitionDuration: 700
      });
      return;
    }
    focusMapOnPoint(lat, lon, MIN_SELECTION_ZOOM);
  };

  const tooltip = (info: PickingInfo): { html: string } | null => {
    const selected = normalizePickedEvent(info.object);
    if (!selected) {
      return null;
    }

    return {
      html: `
        <div style="font-family:Inter,sans-serif;padding:2px;">
          <div style="font-size:13px;font-weight:700;color:#f97316;margin-bottom:4px;">Fire Event</div>
          <div style="font-size:12px;color:#e5e7eb;line-height:1.45;">
            <b>Event ID:</b> ${String(selected.event_id || "unknown")}<br/>
            <b>Cluster events:</b> ${String(selected.cluster_event_count || 1)}<br/>
            <b>Window:</b> ${String(selected.start_time || "n/a")} → ${String(selected.end_time || "n/a")}<br/>
            <b>Sensor:</b> ${String(selected.sensor || "unknown")}<br/>
            <b>Detections:</b> ${String(selected.detection_count || 0)}<br/>
            <b>Event score:</b> ${String(selected.event_score || "n/a")}<br/>
            <b>Decision:</b> ${String(selected.denoiser_decision || "unknown")}<br/>
            <b>Review required:</b> ${String(Boolean(selected.review_required))}<br/>
            <b>Perimeter:</b> ${geometryProvenanceLabel(selected)}
          </div>
        </div>
      `
    };
  };

  return { onClick, tooltip };
}
