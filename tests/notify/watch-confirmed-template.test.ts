/**
 * Stage 9 — watch-confirmed email template snapshot tests.
 *
 * The body shape is part of the user-visible contract; small edits should
 * fail loudly so reviewers see the diff.
 */
import { describe, expect, it } from "vitest";
import { renderWatchConfirmedEmail } from "@/lib/notify/watch-confirmed-template";

describe("renderWatchConfirmedEmail", () => {
  const baseArgs = {
    aoiName: "Spring Creek Preserve",
    regionBucket: "5x5:W125_N35",
    areaHa: 850.5,
    firstPollAt: new Date("2026-05-07T14:30:00Z"),
    aoiUrl: "http://localhost:3000/dashboard/aoi/abc-123",
  };

  it("subject is exactly 'Now watching {aoiName}'", () => {
    const r = renderWatchConfirmedEmail(baseArgs);
    expect(r.subject).toBe("Now watching Spring Creek Preserve");
  });

  it("body contains AOI name, area, region, firstPollAt, and aoiUrl", () => {
    const r = renderWatchConfirmedEmail(baseArgs);
    expect(r.markdown).toContain("Spring Creek Preserve");
    expect(r.markdown).toContain("851 ha"); // rounded
    expect(r.markdown).toContain("125°W 35°N");
    expect(r.markdown).toContain("2026-05-07 14:30 UTC");
    expect(r.markdown).toContain("http://localhost:3000/dashboard/aoi/abc-123");
    expect(r.markdown).toContain("/rules");
    expect(r.markdown).toContain("Wildfire Nowcast");
  });

  it("small areas use one decimal of precision", () => {
    const r = renderWatchConfirmedEmail({ ...baseArgs, areaHa: 12.34 });
    expect(r.markdown).toContain("12.3 ha");
  });
});
