import { getJson, postJson } from "./http";

export interface ArchiveAvailabilityResponse {
  has_data: boolean;
  detection_count: number;
}

export interface ArchiveIngestResponse {
  job_id: string;
  estimated_minutes: number;
}

export interface ArchiveIngestStatusResponse {
  status: string;
  error: string | null;
}

export type ArchiveDayIngestStatus = 'queued' | 'started' | 'finished' | 'failed';
export type ArchiveRangeOverallStatus = 'queued' | 'in_progress' | 'completed' | 'partial_failure' | 'not_found';

export interface ArchiveRangeDayStatus {
  date: string;
  status: ArchiveDayIngestStatus;
  error: string | null;
}

export interface ArchiveRangeIngestResponse {
  range_job_id: string;
  dates: string[];
  estimated_minutes: number;
  warning: string | null;
}

export interface ArchiveRangeStatusResponse {
  range_job_id: string;
  days: ArchiveRangeDayStatus[];
  overall_status: ArchiveRangeOverallStatus;
  completed_count: number;
  total_count: number;
}

export async function checkArchiveAvailability(
  date: string,
  timeframe: string
): Promise<ArchiveAvailabilityResponse> {
  return getJson<ArchiveAvailabilityResponse>("/fires/archive/availability", { date, timeframe });
}

export async function triggerArchiveIngest(
  date: string,
  timeframe: string
): Promise<ArchiveIngestResponse> {
  return postJson<ArchiveIngestResponse>("/fires/archive/ingest", { date, timeframe }, 202);
}

export async function getArchiveIngestStatus(jobId: string): Promise<ArchiveIngestStatusResponse> {
  return getJson<ArchiveIngestStatusResponse>(`/fires/archive/ingest/${jobId}`, {});
}

export async function triggerArchiveRangeIngest(
  startDate: string,
  endDate: string
): Promise<ArchiveRangeIngestResponse> {
  return postJson<ArchiveRangeIngestResponse>(
    "/fires/archive/ingest-range",
    { start_date: startDate, end_date: endDate },
    202
  );
}

export async function getArchiveRangeStatus(rangeJobId: string): Promise<ArchiveRangeStatusResponse> {
  return getJson<ArchiveRangeStatusResponse>(`/fires/archive/ingest-range/${rangeJobId}/status`, {});
}
