import type {
  ActiveModelsResponse,
  JitCreateResponse,
  JitForecastStatus
} from "../types/api";
import { ApiError, getJson, isoFormat, postJson } from "./http";
import type { BBox } from "../types/api";

export async function getJitForecastStatus(jobId: string): Promise<JitForecastStatus> {
  return getJson<JitForecastStatus>(`/forecast/jit/${jobId}`, {});
}

export async function getActiveSpreadModelId(): Promise<string> {
  const payload = await getJson<ActiveModelsResponse>("/internal/models/active", {});
  const modelId = payload.models?.spread?.model_id;
  if (modelId && modelId.trim().length > 0) {
    return modelId.trim();
  }
  throw new ApiError("No active spread model is promoted. Promote a spread model and retry.", {
    statusCode: 422
  });
}

export async function createJitForecast(args: {
  bbox: BBox;
  forecastReferenceTime: Date;
  horizonsHours: number[];
  modelId: string;
}): Promise<JitCreateResponse> {
  return postJson<JitCreateResponse>(
    "/forecast/jit",
    {
      bbox: args.bbox,
      forecast_reference_time: isoFormat(args.forecastReferenceTime),
      horizons_hours: args.horizonsHours,
      model_id: args.modelId
    },
    202
  );
}

export async function createJitForecastFromFront(args: {
  frontId: string;
  bufferKm: number;
  forecastReferenceTime: Date;
  horizonsHours: number[];
  modelId: string;
}): Promise<JitCreateResponse> {
  return postJson<JitCreateResponse>(
    "/forecast/jit/from-front",
    {
      front_id: args.frontId,
      buffer_km: args.bufferKm,
      forecast_reference_time: isoFormat(args.forecastReferenceTime),
      horizons_hours: args.horizonsHours,
      model_id: args.modelId
    },
    202
  );
}
