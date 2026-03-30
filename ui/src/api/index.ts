// HTTP primitives and error classes
export { ApiError, ApiUnavailableError, getJson, postJson, putJson, deleteRequest, isoFormat, toSearchParams, withTimeout } from "./http";

// Fire / front / risk / geocode
export {
  getFireEvents,
  getFireFronts,
  getReverseGeocode,
  getRiskGrid,
  getWeatherForPoint,
  buildEventKey,
  buildFiresCsvExportUrl,
  buildMapPngExportUrl
} from "./fires";

// Archive
export type {
  ArchiveAvailabilityResponse,
  ArchiveIngestResponse,
  ArchiveIngestStatusResponse,
  ArchiveDayIngestStatus,
  ArchiveRangeOverallStatus,
  ArchiveRangeDayStatus,
  ArchiveRangeIngestResponse,
  ArchiveRangeStatusResponse
} from "./archive";
export {
  checkArchiveAvailability,
  triggerArchiveIngest,
  getArchiveIngestStatus,
  triggerArchiveRangeIngest,
  getArchiveRangeStatus
} from "./archive";

// Forecast
export {
  getJitForecastStatus,
  getActiveSpreadModelId,
  createJitForecast,
  createJitForecastFromFront
} from "./forecast";

// Review queue
export {
  getDenoiserReviewQueue,
  resolveDenoiserReviewItem,
  getReviewEventDetail,
} from "./review";

// AOI / Watchlist / Export / Data freshness
export {
  getDataFreshnessStatus,
  listAOIs,
  getAOI,
  createAOI,
  deleteAOI,
  configureAOIWatch,
  getWatchlist
} from "./watchlist";
