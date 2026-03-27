import { beforeEach, describe, expect, it } from "vitest";

import { useAppStore } from "../state/store";
import { computeFullDayTimeRange } from "../utils/time";

// ---------------------------------------------------------------------------
// computeFullDayTimeRange
// ---------------------------------------------------------------------------

describe("computeFullDayTimeRange", () => {
  it("returns UTC midnight to 23:59:59 for a given date", () => {
    const { startTime, endTime } = computeFullDayTimeRange("2026-03-20");
    expect(startTime.toISOString()).toBe("2026-03-20T00:00:00.000Z");
    expect(endTime.toISOString()).toBe("2026-03-20T23:59:59.000Z");
  });

  it("handles month and year boundaries correctly", () => {
    const { startTime, endTime } = computeFullDayTimeRange("2025-12-31");
    expect(startTime.toISOString()).toBe("2025-12-31T00:00:00.000Z");
    expect(endTime.toISOString()).toBe("2025-12-31T23:59:59.000Z");
  });

  it("startTime is strictly before endTime", () => {
    const { startTime, endTime } = computeFullDayTimeRange("2026-01-15");
    expect(startTime.getTime()).toBeLessThan(endTime.getTime());
  });

  it("start and end are on the same calendar date", () => {
    const { startTime, endTime } = computeFullDayTimeRange("2026-06-01");
    expect(startTime.toISOString().slice(0, 10)).toBe("2026-06-01");
    expect(endTime.toISOString().slice(0, 10)).toBe("2026-06-01");
  });
});

// ---------------------------------------------------------------------------
// Archive range store actions
// ---------------------------------------------------------------------------

describe("archive range store", () => {
  const baseState = useAppStore.getState();

  beforeEach(() => {
    useAppStore.setState({
      ...baseState,
      archive: {
        viewMode: "live",
        archiveDate: null,
        archiveTimeframe: null,
        archiveSubMode: "single",
        rangeStart: null,
        rangeEnd: null,
        rangeJobId: null,
        scrubDate: null,
      },
    });
  });

  it("setArchiveSubMode switches between single and range", () => {
    useAppStore.getState().setArchiveSubMode("range");
    expect(useAppStore.getState().archive.archiveSubMode).toBe("range");

    useAppStore.getState().setArchiveSubMode("single");
    expect(useAppStore.getState().archive.archiveSubMode).toBe("single");
  });

  it("setArchiveRange enters archive/range mode with correct dates", () => {
    useAppStore.getState().setArchiveRange("2026-03-18", "2026-03-20");
    const { archive } = useAppStore.getState();

    expect(archive.viewMode).toBe("archive");
    expect(archive.archiveSubMode).toBe("range");
    expect(archive.rangeStart).toBe("2026-03-18");
    expect(archive.rangeEnd).toBe("2026-03-20");
    expect(archive.scrubDate).toBe("2026-03-18");  // initialised to start
    expect(archive.rangeJobId).toBeNull();          // job not yet assigned
  });

  it("setArchiveRange clears selection", () => {
    useAppStore.setState({ ...useAppStore.getState(), selectedEvent: {} as never, lastClick: {} as never });
    useAppStore.getState().setArchiveRange("2026-03-18", "2026-03-20");
    expect(useAppStore.getState().selectedEvent).toBeNull();
    expect(useAppStore.getState().lastClick).toBeNull();
  });

  it("setRangeJobId stores the job id", () => {
    const id = "test-range-job-uuid";
    useAppStore.getState().setRangeJobId(id);
    expect(useAppStore.getState().archive.rangeJobId).toBe(id);
  });

  it("setRangeJobId accepts null to clear", () => {
    useAppStore.getState().setRangeJobId("something");
    useAppStore.getState().setRangeJobId(null);
    expect(useAppStore.getState().archive.rangeJobId).toBeNull();
  });

  it("setScrubDate advances the current viewing date", () => {
    useAppStore.getState().setArchiveRange("2026-03-18", "2026-03-20");
    useAppStore.getState().setScrubDate("2026-03-19");
    expect(useAppStore.getState().archive.scrubDate).toBe("2026-03-19");
  });

  it("setScrubDate clears selection and lastClick", () => {
    useAppStore.setState({ ...useAppStore.getState(), selectedEvent: {} as never, lastClick: {} as never });
    useAppStore.getState().setScrubDate("2026-03-19");
    expect(useAppStore.getState().selectedEvent).toBeNull();
    expect(useAppStore.getState().lastClick).toBeNull();
  });

  it("exitToLiveMode resets all range fields", () => {
    useAppStore.getState().setArchiveRange("2026-03-18", "2026-03-20");
    useAppStore.getState().setRangeJobId("some-job");
    useAppStore.getState().setScrubDate("2026-03-19");
    useAppStore.getState().exitToLiveMode();

    const { archive } = useAppStore.getState();
    expect(archive.viewMode).toBe("live");
    expect(archive.archiveSubMode).toBe("single");
    expect(archive.rangeStart).toBeNull();
    expect(archive.rangeEnd).toBeNull();
    expect(archive.rangeJobId).toBeNull();
    expect(archive.scrubDate).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Date range enumeration (mirrors the backend's date list logic)
// ---------------------------------------------------------------------------

describe("range date enumeration", () => {
  function datesInRange(start: string, end: string): string[] {
    const dates: string[] = [];
    const startMs = Date.parse(start);
    const endMs = Date.parse(end);
    for (let ms = startMs; ms <= endMs; ms += 86_400_000) {
      dates.push(new Date(ms).toISOString().slice(0, 10));
    }
    return dates;
  }

  it("generates a single date for same start and end", () => {
    expect(datesInRange("2026-03-20", "2026-03-20")).toEqual(["2026-03-20"]);
  });

  it("generates all dates inclusive", () => {
    const dates = datesInRange("2026-03-18", "2026-03-20");
    expect(dates).toEqual(["2026-03-18", "2026-03-19", "2026-03-20"]);
  });

  it("handles month crossings", () => {
    const dates = datesInRange("2026-01-30", "2026-02-02");
    expect(dates).toEqual(["2026-01-30", "2026-01-31", "2026-02-01", "2026-02-02"]);
  });

  it("length matches (end - start) + 1 days", () => {
    const dates = datesInRange("2026-03-15", "2026-03-21");
    expect(dates).toHaveLength(7);
  });
});
