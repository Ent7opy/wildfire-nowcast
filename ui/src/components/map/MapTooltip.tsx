import type { PickingInfo } from "@deck.gl/core";
import { normalizePickedEvent } from "../../utils/selection";
import { geometryProvenanceLabel } from "../fire-details/types";

export function buildTooltipContent(info: PickingInfo): { html: string } | null {
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
}
