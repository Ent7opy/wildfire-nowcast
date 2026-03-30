import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import { windCompassLabel, rhRiskDisplay, WeatherBlock } from "../components/fire-details/WeatherBlock";
import type { WeatherContext } from "../types/api";

// ---------------------------------------------------------------------------
// Helper unit tests
// ---------------------------------------------------------------------------

describe("windCompassLabel", () => {
  it.each([
    [0, "N"],
    [45, "NE"],
    [90, "E"],
    [135, "SE"],
    [180, "S"],
    [225, "SW"],
    [270, "W"],
    [315, "NW"],
    [350, "N"],       // wraps back to North
    [230, "SW"],
    [22, "NNE"],
  ])("converts %d° to %s", (deg, expected) => {
    expect(windCompassLabel(deg)).toBe(expected);
  });
});

describe("rhRiskDisplay", () => {
  it("returns Critical styling for critical risk", () => {
    const d = rhRiskDisplay("critical");
    expect(d.label).toBe("Critical");
  });

  it("returns Elevated styling for elevated risk", () => {
    const d = rhRiskDisplay("elevated");
    expect(d.label).toBe("Elevated");
  });

  it("returns Normal styling for normal risk", () => {
    const d = rhRiskDisplay("normal");
    expect(d.label).toBe("Normal");
  });
});

// ---------------------------------------------------------------------------
// Component render tests
// ---------------------------------------------------------------------------

function makeWeather(overrides: Partial<WeatherContext> = {}): WeatherContext {
  return {
    wind_speed_ms: 12.4,
    wind_direction_deg: 230,
    relative_humidity_pct: 18,
    rh_fire_risk: "critical",
    temperature_c: 36.2,
    precip_mm_24h: 0.0,
    source_run_time: "2026-03-28T06:00:00Z",
    data_age_hours: 2.1,
    resolution_note: "GFS 0.25\u00b0 \u2014 nearest grid point (~25 km)",
    bias_correction: {
      applied: true,
      method: "affine (fitted against ERA5 reanalysis)",
      variables: ["u10", "v10", "t2m", "rh2m"],
    },
    ...overrides,
  };
}

afterEach(cleanup);

describe("WeatherBlock", () => {
  it("renders weather data with wind, humidity, temperature, and precip", () => {
    const weather = makeWeather();
    render(<WeatherBlock weather={weather} unavailableReason={null} />);

    expect(screen.getByTestId("weather-block")).toBeDefined();
    expect(screen.getByText("12.4 m/s")).toBeDefined();
    expect(screen.getByText("36.2 °C")).toBeDefined();
    expect(screen.getByText("0.0 mm")).toBeDefined();
    expect(screen.getByText("18%")).toBeDefined();
  });

  it("renders human-readable wind direction with compass label", () => {
    const weather = makeWeather({ wind_direction_deg: 230 });
    render(<WeatherBlock weather={weather} unavailableReason={null} />);

    const windDir = screen.getByTestId("wind-direction");
    expect(windDir.textContent).toContain("SW");
    expect(windDir.textContent).toContain("230°");
  });

  it("shows critical RH fire risk label for low humidity", () => {
    const weather = makeWeather({ relative_humidity_pct: 12, rh_fire_risk: "critical" });
    render(<WeatherBlock weather={weather} unavailableReason={null} />);

    const label = screen.getByTestId("rh-risk-label");
    expect(label.textContent).toContain("Critical");
    expect(label.textContent).toContain("fire risk");
  });

  it("shows elevated RH fire risk label", () => {
    const weather = makeWeather({ relative_humidity_pct: 22, rh_fire_risk: "elevated" });
    render(<WeatherBlock weather={weather} unavailableReason={null} />);

    const label = screen.getByTestId("rh-risk-label");
    expect(label.textContent).toContain("Elevated");
  });

  it("shows normal RH fire risk label for high humidity", () => {
    const weather = makeWeather({ relative_humidity_pct: 55, rh_fire_risk: "normal" });
    render(<WeatherBlock weather={weather} unavailableReason={null} />);

    const label = screen.getByTestId("rh-risk-label");
    expect(label.textContent).toContain("Normal");
  });

  it("shows data provenance inline — age, resolution, bias correction", () => {
    const weather = makeWeather();
    render(<WeatherBlock weather={weather} unavailableReason={null} />);

    const provenance = screen.getByTestId("weather-provenance");
    expect(provenance.textContent).toContain("GFS 0.25");
    expect(provenance.textContent).toContain("2.1h");
    expect(provenance.textContent).toContain("Bias-corrected");
  });

  it("omits bias correction note when not applied", () => {
    const weather = makeWeather({
      bias_correction: { applied: false, method: "", variables: [] },
    });
    render(<WeatherBlock weather={weather} unavailableReason={null} />);

    const provenance = screen.getByTestId("weather-provenance");
    expect(provenance.textContent).not.toContain("Bias-corrected");
  });

  it("renders null state with reason string from API", () => {
    render(
      <WeatherBlock
        weather={null}
        unavailableReason="No GFS weather run covers this location within the tolerance window"
      />
    );

    expect(screen.getByText(/No GFS weather run covers/)).toBeDefined();
    expect(screen.queryByTestId("weather-block")).toBeNull();
  });

  it("renders null state with default message when no reason provided", () => {
    render(<WeatherBlock weather={null} unavailableReason={null} />);

    expect(screen.getByText(/Weather data not available/)).toBeDefined();
  });

  it("renders loading state", () => {
    render(<WeatherBlock weather={null} unavailableReason={null} isLoading />);

    expect(screen.getByText(/Loading weather data/)).toBeDefined();
  });
});
