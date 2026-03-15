import { describe, expect, it } from "vitest";

import { comparePriorityFeedEvents } from "../utils/priorityFeed";
import type { FireEvent } from "../types/api";

function sortEvents(events: FireEvent[]): FireEvent[] {
  return [...events].sort(comparePriorityFeedEvents);
}

describe("priority feed ranking", () => {
  it("sorts events by FRP descending", () => {
    const events: FireEvent[] = [
      { event_id: "a", frp_max: 6.75, event_score: 0.9 },
      { event_id: "b", frp_max: 10.17, event_score: 0.2 },
      { event_id: "c", frp_max: 7.03, event_score: 0.8 }
    ];

    const sorted = sortEvents(events);
    expect(sorted.map((event) => event.event_id)).toEqual(["b", "c", "a"]);
  });

  it("keeps FRP-ranked events ahead of brightness-only events", () => {
    const events: FireEvent[] = [
      { event_id: "a", brightness_max: 420.2 },
      { event_id: "b", frp_mean: 2.1 },
      { event_id: "c", frp_max: 1.2 }
    ];

    const sorted = sortEvents(events);
    expect(sorted.map((event) => event.event_id)).toEqual(["c", "b", "a"]);
  });

  it("uses score and detections only as tie-breakers when intensity matches", () => {
    const events: FireEvent[] = [
      { event_id: "a", frp_max: 8, event_score: 0.4, detection_count: 20 },
      { event_id: "b", frp_max: 8, event_score: 0.8, detection_count: 10 },
      { event_id: "c", frp_max: 8, event_score: 0.8, detection_count: 25 }
    ];

    const sorted = sortEvents(events);
    expect(sorted.map((event) => event.event_id)).toEqual(["c", "b", "a"]);
  });
});
