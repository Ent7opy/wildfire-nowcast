import type { ArchiveTimeframe, FiltersState } from "../types/state";

export interface TimeframeDef {
  id: ArchiveTimeframe;
  label: string;
  hours: [number, number];  // [startHour, endHour] inclusive, local time
}

export const TIMEFRAME_DEFS: TimeframeDef[] = [
  { id: 'morning',   label: 'Morning',   hours: [6,  11] },
  { id: 'afternoon', label: 'Afternoon', hours: [12, 17] },
  { id: 'evening',   label: 'Evening',   hours: [18, 23] },
  { id: 'night',     label: 'Night',     hours: [0,  5]  },
];

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

export function computeArchiveTimeRange(date: string, timeframe: ArchiveTimeframe): { startTime: Date; endTime: Date } {
  const def = TIMEFRAME_DEFS.find((d) => d.id === timeframe) ?? TIMEFRAME_DEFS[1];
  const [startHour, endHour] = def.hours;
  // Anchor to UTC — satellite acquisition timestamps (acq_time) are always UTC,
  // and the backend's _timeframe_window also treats these hours as UTC.
  // Using Date.UTC avoids browser-timezone drift shifting the query window.
  const [year, month, day] = date.split('-').map(Number);
  const startTime = new Date(Date.UTC(year, month - 1, day, startHour, 0, 0, 0));
  const endTime = new Date(Date.UTC(year, month - 1, day, endHour, 59, 59, 0));
  return { startTime, endTime };
}

export function currentTimeframe(): ArchiveTimeframe {
  const hour = new Date().getHours();
  for (const def of TIMEFRAME_DEFS) {
    const [s, e] = def.hours;
    if (s <= e ? hour >= s && hour <= e : hour >= s || hour <= e) {
      return def.id;
    }
  }
  return 'afternoon';
}

/** Return every calendar date (YYYY-MM-DD) from start to end, inclusive. */
export function datesInRange(start: string, end: string): string[] {
  const dates: string[] = [];
  const startMs = Date.parse(start);
  const endMs = Date.parse(end);
  for (let ms = startMs; ms <= endMs; ms += 86_400_000) {
    dates.push(new Date(ms).toISOString().slice(0, 10));
  }
  return dates;
}

export function computeFullDayTimeRange(date: string): { startTime: Date; endTime: Date } {
  const [year, month, day] = date.split('-').map(Number);
  const startTime = new Date(Date.UTC(year, month - 1, day, 0, 0, 0, 0));
  const endTime = new Date(Date.UTC(year, month - 1, day, 23, 59, 59, 0));
  return { startTime, endTime };
}

export function formatTimeWindow(filters: FiltersState): string {
  const hours = filters.hoursStart - filters.hoursEnd;
  if (filters.hoursEnd === 0) {
    return hours === 1 ? "Last 1 hour" : `Last ${hours} hours`;
  }
  return `${hours}h window (${filters.hoursStart}h ago to ${filters.hoursEnd}h ago)`;
}
