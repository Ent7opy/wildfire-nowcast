import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { useUrlStateSync } from "../hooks/useUrlStateSync";
import { useAppStore } from "../state/store";

function TestComponent(): null {
  useUrlStateSync();
  return null;
}

describe("URL sync", () => {
  it("hydrates filters from URL parameters", async () => {
    window.history.replaceState(null, "", "/?start=6&end=1&likelihood=0.6&active_only=false&cluster=true");

    useAppStore.setState({
      ...useAppStore.getState(),
      initializedFromUrl: false
    });

    render(<TestComponent />);

    const state = useAppStore.getState();
    expect(state.filters.hoursStart).toBe(6);
    expect(state.filters.hoursEnd).toBe(1);
    expect(state.filters.minLikelihood).toBe(0.6);
    expect(state.filters.activeOnly).toBe(false);
    expect(state.filters.clusterPoints).toBe(true);
  });

  it("writes canonical filters to URL", () => {
    window.history.replaceState(null, "", "/");

    useAppStore.setState({
      ...useAppStore.getState(),
      initializedFromUrl: true,
      filters: {
        hoursStart: 24,
        hoursEnd: 0,
        minLikelihood: 0.33,
        activeOnly: true,
        clusterPoints: false
      },
      activePreset: "Custom"
    });

    render(<TestComponent />);

    const params = new URLSearchParams(window.location.search);
    expect(params.get("start")).toBe("24");
    expect(params.get("end")).toBe("0");
    expect(params.get("likelihood")).toBe("0.33");
    expect(params.get("active_only")).toBe("true");
    expect(params.get("cluster")).toBe("false");
    expect(params.get("preset")).toBe("Custom");
  });
});
