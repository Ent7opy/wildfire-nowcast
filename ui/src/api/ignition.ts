import type { BBox, IgnitionGridResponse } from "../types/api";
import type { IgnitionHorizon } from "../types/state";
import { getJson } from "./http";

const HORIZON_PARAM: Record<IgnitionHorizon, string> = {
  'now': 'now',
  '+24h': '24h',
  '+48h': '48h',
};

export async function getIgnitionGrid(args: {
  bbox: BBox;
  horizon: IgnitionHorizon;
}): Promise<IgnitionGridResponse> {
  const [minLon, minLat, maxLon, maxLat] = args.bbox;
  return getJson<IgnitionGridResponse>('/ignition', {
    min_lon: minLon,
    min_lat: minLat,
    max_lon: maxLon,
    max_lat: maxLat,
    horizon: HORIZON_PARAM[args.horizon],
  });
}
