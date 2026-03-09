import type { FiltersState } from "../types/state";

export function isoFormat(date: Date): string {
  const d = new Date(date);
  d.setMilliseconds(0);
  if (d.getTimezoneOffset() === 0) {
    return d.toISOString().replace(".000", "");
  }
  return d.toISOString();
}

export function computeTimeRange(filters: FiltersState): { startTime: Date; endTime: Date } {
  const now = new Date();
  now.setSeconds(0, 0);
  const endTime = new Date(now.getTime() - filters.hoursEnd * 3600_000);
  const spanHours = filters.hoursStart - filters.hoursEnd;
  const startTime = new Date(endTime.getTime() - spanHours * 3600_000);
  return { startTime, endTime };
}

export function formatTimeWindow(filters: FiltersState): string {
  const hours = filters.hoursStart - filters.hoursEnd;
  if (filters.hoursEnd === 0) {
    return hours === 1 ? "Last 1 hour" : `Last ${hours} hours`;
  }
  return `${hours}h window (${filters.hoursStart}h ago to ${filters.hoursEnd}h ago)`;
}
