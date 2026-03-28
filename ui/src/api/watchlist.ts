import type {
  AOI,
  AOIListResponse,
  DataFreshnessResponse,
  WatchConfigRequest,
  WatchlistResponse
} from "../types/api";
import { deleteRequest, getJson, postJson, putJson } from "./http";

export async function getDataFreshnessStatus(): Promise<DataFreshnessResponse> {
  return getJson<DataFreshnessResponse>("/health/data-freshness", {}, { slowPath: true });
}

export async function listAOIs(params?: { limit?: number; offset?: number; q?: string }): Promise<AOIListResponse> {
  return getJson<AOIListResponse>("/aois", params ?? {});
}

export async function getAOI(aoiId: string): Promise<AOI> {
  return getJson<AOI>(`/aois/${aoiId}`, {});
}

export async function createAOI(payload: {
  name: string;
  geometry: Record<string, unknown>;
  description?: string;
  tags?: Record<string, unknown>;
}): Promise<AOI> {
  return postJson<AOI>("/aois", payload as Record<string, unknown>, 201);
}

export async function deleteAOI(aoiId: string): Promise<void> {
  return deleteRequest(`/aois/${aoiId}`);
}

export async function configureAOIWatch(aoiId: string, config: WatchConfigRequest): Promise<AOI> {
  return putJson<AOI>(`/aois/${aoiId}/watch`, { ...config });
}

export async function getWatchlist(): Promise<WatchlistResponse> {
  return getJson<WatchlistResponse>("/aois/watchlist", {});
}
