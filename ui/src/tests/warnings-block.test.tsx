import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import { WarningsBlock } from "../components/fire-details/WarningsBlock";
import type { WeatherWarningBrief } from "../types/api";

afterEach(cleanup);

function makeWarning(overrides: Partial<WeatherWarningBrief> = {}): WeatherWarningBrief {
  return {
    source: "meteoalarm",
    warning_type: "wind",
    severity: "red",
    headline: "Extreme wind warning",
    expires: new Date(Date.now() + 3 * 60 * 60 * 1000).toISOString(), // +3h
    country_code: "GR",
    ...overrides,
  };
}

describe("WarningsBlock", () => {
  it("renders null when warnings is null", () => {
    const { container } = render(<WarningsBlock warnings={null} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders null when warnings is empty array", () => {
    const { container } = render(<WarningsBlock warnings={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders block with a warning", () => {
    render(<WarningsBlock warnings={[makeWarning()]} />);
    expect(screen.getByTestId("warnings-block")).toBeDefined();
    expect(screen.getByText("Extreme wind warning")).toBeDefined();
  });

  it("renders severity badge for red warning", () => {
    render(<WarningsBlock warnings={[makeWarning({ severity: "red" })]} />);
    const badge = screen.getByTestId("warning-severity-badge-0");
    expect(badge.textContent).toBe("RED");
  });

  it("renders severity badge for orange warning", () => {
    render(<WarningsBlock warnings={[makeWarning({ severity: "orange" })]} />);
    const badge = screen.getByTestId("warning-severity-badge-0");
    expect(badge.textContent).toBe("ORANGE");
  });

  it("renders multiple warnings sorted by severity (red first)", () => {
    const warnings = [
      makeWarning({ severity: "yellow", headline: "Yellow warning" }),
      makeWarning({ severity: "red", headline: "Red warning" }),
      makeWarning({ severity: "orange", headline: "Orange warning" }),
    ];
    render(<WarningsBlock warnings={warnings} />);

    const badges = screen.getAllByTestId(/warning-severity-badge-/);
    expect(badges[0].textContent).toBe("RED");
    expect(badges[1].textContent).toBe("ORANGE");
    expect(badges[2].textContent).toBe("YELLOW");
  });

  it("shows country code when present", () => {
    render(<WarningsBlock warnings={[makeWarning({ country_code: "GR" })]} />);
    expect(screen.getByText("GR")).toBeDefined();
  });

  it("shows source attribution", () => {
    render(<WarningsBlock warnings={[makeWarning()]} />);
    expect(screen.getByText(/MeteoAlarm/)).toBeDefined();
  });

  it("renders correct warning type label for drought", () => {
    render(<WarningsBlock warnings={[makeWarning({ warning_type: "drought" })]} />);
    expect(screen.getByText(/Drought/)).toBeDefined();
  });

  it("renders time remaining for non-expired warning", () => {
    render(<WarningsBlock warnings={[makeWarning()]} />);
    expect(screen.getByText(/remaining/)).toBeDefined();
  });
});
