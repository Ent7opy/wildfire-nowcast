import { describe, expect, it } from "vitest";

import { forecastButtonState } from "../utils/forecast";

describe("forecast button guards", () => {
  it("disables while forecast is running", () => {
    const state = forecastButtonState({ forecastRunning: true, sameEventCompleted: false });
    expect(state.disabled).toBe(true);
    expect(state.label).toContain("Generating");
  });

  it("disables when same event already completed", () => {
    const state = forecastButtonState({ forecastRunning: false, sameEventCompleted: true });
    expect(state.disabled).toBe(true);
    expect(state.label).toContain("Already Generated");
  });

  it("enables for new events", () => {
    const state = forecastButtonState({ forecastRunning: false, sameEventCompleted: false });
    expect(state.disabled).toBe(false);
    expect(state.label).toBe("Generate Spread Forecast");
  });
});
