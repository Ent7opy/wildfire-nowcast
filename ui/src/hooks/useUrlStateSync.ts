import { useEffect } from "react";

import { useAppStore } from "../state/store";

export function useUrlStateSync(): void {
  const initializeFromUrl = useAppStore((s) => s.initializeFromUrl);
  const initialized = useAppStore((s) => s.initializedFromUrl);
  const filters = useAppStore((s) => s.filters);
  const activePreset = useAppStore((s) => s.activePreset);

  useEffect(() => {
    initializeFromUrl();
  }, [initializeFromUrl]);

  useEffect(() => {
    if (!initialized || typeof window === "undefined") {
      return;
    }

    const params = new URLSearchParams(window.location.search);
    params.set("start", String(filters.hoursStart));
    params.set("end", String(filters.hoursEnd));
    params.set("likelihood", filters.minLikelihood.toFixed(2));
    params.set("active_only", filters.activeOnly ? "true" : "false");
    params.set("cluster", filters.clusterPoints ? "true" : "false");

    if (activePreset) {
      params.set("preset", activePreset);
    } else {
      params.delete("preset");
    }

    const next = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState(null, "", next);
  }, [initialized, filters, activePreset]);
}
