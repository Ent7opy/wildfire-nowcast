import { useMemo } from "react";
import type { FireEvent } from "../types/api";
import type { AssistantConfidenceFilter, AssistantViewEventSummary } from "../types/state";
import type { RegionFilterValue } from "../utils/continents";
import { matchesRegionFilter } from "../utils/continents";
import { toFiniteNumber } from "../utils/priorityFeed";
import { HIGH_CONFIDENCE_THRESHOLD } from "../components/map/layers/layerConfig";

function isHighConfidence(event: FireEvent): boolean {
  const score = toFiniteNumber(event.event_score);
  return score !== null && score >= HIGH_CONFIDENCE_THRESHOLD;
}

function locationLabel(event: FireEvent): string {
  const candidates = [
    event.location_name,
    event.region_name,
    event.admin1_name,
    event.admin0_name,
    event.country
  ];
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim().length > 0) {
      return candidate.trim();
    }
  }
  const lat = toFiniteNumber(event.lat);
  const lon = toFiniteNumber(event.lon);
  if (lat !== null && lon !== null) {
    return `${lat.toFixed(2)}, ${lon.toFixed(2)}`;
  }
  return "Unknown region";
}

function toEventSummary(event: FireEvent): AssistantViewEventSummary {
  const score = toFiniteNumber(event.event_score);
  return {
    eventId: String(event.event_id || "unknown"),
    locationLabel: locationLabel(event),
    lat: toFiniteNumber(event.lat),
    lon: toFiniteNumber(event.lon),
    eventScore: score,
    detectionCount: Number(toFiniteNumber(event.detection_count) || 0),
    frontCount: Number(toFiniteNumber(event.front_count) || 0),
    endTime: (() => {
      if (typeof event.end_time !== "string") return null;
      const trimmed = event.end_time.trim();
      return trimmed.length > 0 ? trimmed : null;
    })(),
    sensor: (() => {
      if (typeof event.sensor !== "string") return null;
      const trimmed = event.sensor.trim();
      return trimmed.length > 0 ? trimmed : null;
    })(),
    source: (() => {
      if (typeof event.source !== "string") return null;
      const trimmed = event.source.trim();
      return trimmed.length > 0 ? trimmed : null;
    })(),
    reviewRequired: Boolean(event.review_required),
    denoiserDecision: (() => {
      if (typeof event.denoiser_decision !== "string") return null;
      const trimmed = event.denoiser_decision.trim();
      return trimmed.length > 0 ? trimmed : null;
    })()
  };
}

interface UseAppDerivedStateParams {
  visibleEvents: FireEvent[];
  confidenceFilter: AssistantConfidenceFilter;
  regionFilter: RegionFilterValue;
  isArchiveMode: boolean;
  safetyProximityRadiusKm?: number;
  safetyUserLocation?: { lat: number; lon: number } | null;
}

export function useAppDerivedState({
  visibleEvents,
  confidenceFilter,
  regionFilter,
  isArchiveMode
}: UseAppDerivedStateParams) {
  const filteredEvents = useMemo(() => {
    return visibleEvents.filter((event) => {
      if (confidenceFilter === "High" && !isHighConfidence(event)) {
        return false;
      }
      return matchesRegionFilter(event, regionFilter);
    });
  }, [confidenceFilter, regionFilter, visibleEvents]);

  const safetyEvents = useMemo(
    () => (isArchiveMode ? [] : visibleEvents),
    [isArchiveMode, visibleEvents]
  );

  const totalDetections = useMemo(() => {
    const aggregate = filteredEvents.reduce((sum, event) => sum + (toFiniteNumber(event.detection_count) || 0), 0);
    return aggregate > 0 ? aggregate : filteredEvents.length;
  }, [filteredEvents]);

  const activePerimeters = useMemo(() => {
    return filteredEvents.reduce((sum, event) => sum + (toFiniteNumber(event.front_count) || 0), 0);
  }, [filteredEvents]);

  const averageScore = useMemo(() => {
    const scores = filteredEvents
      .map((event) => toFiniteNumber(event.event_score))
      .filter((score): score is number => score !== null);
    if (scores.length === 0) return null;
    return scores.reduce((sum, score) => sum + score, 0) / scores.length;
  }, [filteredEvents]);

  const confidencePercent = useMemo(() => {
    if (filteredEvents.length === 0) return null;
    const highCount = filteredEvents.filter((event) => isHighConfidence(event)).length;
    return (highCount / filteredEvents.length) * 100;
  }, [filteredEvents]);

  const maxIntensityFrp = useMemo(() => {
    if (filteredEvents.length === 0) return null;
    const frpValues = filteredEvents
      .map((e) => toFiniteNumber(e.frp_max))
      .filter((v): v is number => v !== null);
    return frpValues.length > 0 ? Math.max(...frpValues) : null;
  }, [filteredEvents]);

  const topEventsForAssistant = useMemo(() => {
    return [...filteredEvents]
      .sort((a, b) => {
        const scoreDiff = (toFiniteNumber(b.event_score) || 0) - (toFiniteNumber(a.event_score) || 0);
        if (scoreDiff !== 0) return scoreDiff;
        return (toFiniteNumber(b.detection_count) || 0) - (toFiniteNumber(a.detection_count) || 0);
      })
      .slice(0, 20)
      .map(toEventSummary);
  }, [filteredEvents]);

  return {
    filteredEvents,
    safetyEvents,
    totalDetections,
    activePerimeters,
    averageScore,
    confidencePercent,
    maxIntensityFrp,
    topEventsForAssistant
  };
}
