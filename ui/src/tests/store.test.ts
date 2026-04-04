import { beforeEach, describe, expect, it } from "vitest";

import { useAppStore } from "../state/store";

const baseState = useAppStore.getState();

describe("app store filters", () => {
  beforeEach(() => {
    useAppStore.setState({
      ...baseState,
      filters: {
        hoursStart: 6,
        hoursEnd: 0,
        minLikelihood: 0,
        activeOnly: true,
        clusterPoints: true
      },
      activePreset: "Last 6h All",
      layers: {
        showFires: true,
        showFronts: true,
        showForecast: true,
        showRisk: false,
        showWarnings: false,
        showIgnition: false,
        basemap: 'dark' as const
      }
    });
  });

  it("keeps hoursStart above hoursEnd", () => {
    useAppStore.getState().setFilters({ hoursStart: 4, hoursEnd: 4 });
    const state = useAppStore.getState();
    expect(state.filters.hoursStart).toBe(5);
    expect(state.filters.hoursEnd).toBe(4);
  });

  it("disables risk layer when clustering is turned off", () => {
    useAppStore.getState().setRiskVisibility(true);
    useAppStore.getState().setFilters({ clusterPoints: false });

    const state = useAppStore.getState();
    expect(state.filters.clusterPoints).toBe(false);
    expect(state.layers.showRisk).toBe(false);
  });
});
