const DEFAULT_API_BASE_URL = "http://localhost:8000";
const DEFAULT_VECTOR_TILES_BASE_URL = "http://localhost:7800";
const DEFAULT_FORECAST_REGION_NAME = "smoke_grid";

function normalizeBaseUrl(value: string | undefined, fallback: string): string {
  const raw = (value || fallback).trim();
  return raw.replace(/\/$/, "");
}

function rewriteInternalServiceHost(baseUrl: string): string {
  try {
    const url = new URL(baseUrl);
    if (url.hostname !== "api") {
      return baseUrl;
    }
    url.hostname = "localhost";
    if (!url.port) {
      url.port = "8000";
    }
    return url.toString().replace(/\/$/, "");
  } catch {
    return baseUrl;
  }
}

export function apiBaseUrl(): string {
  return normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL, DEFAULT_API_BASE_URL);
}

export function apiPublicBaseUrl(): string {
  const configured = import.meta.env.VITE_API_PUBLIC_BASE_URL;
  if (configured && configured.trim().length > 0) {
    return rewriteInternalServiceHost(normalizeBaseUrl(configured, DEFAULT_API_BASE_URL));
  }
  return rewriteInternalServiceHost(apiBaseUrl());
}

export function apiBaseUrlCandidates(): string[] {
  const primary = apiBaseUrl();
  const publicBase = apiPublicBaseUrl();
  const out: string[] = [];

  for (const candidate of [publicBase, primary]) {
    if (!candidate || out.includes(candidate)) {
      continue;
    }
    out.push(candidate);
  }

  try {
    const parsed = new URL(primary);
    if (parsed.hostname === "api") {
      const localhost = `${parsed.protocol}//localhost:${parsed.port || "8000"}`;
      const loopback = `${parsed.protocol}//127.0.0.1:${parsed.port || "8000"}`;
      for (const candidate of [localhost, loopback]) {
        if (!out.includes(candidate)) {
          out.push(candidate);
        }
      }
    }
  } catch {
    // ignore invalid URL and just use already-known candidates
  }

  return out;
}

export function vectorTilesBaseUrl(): string {
  return normalizeBaseUrl(import.meta.env.VITE_VECTOR_TILES_PUBLIC_BASE_URL, DEFAULT_VECTOR_TILES_BASE_URL);
}

export function forecastRegionName(): string {
  return (import.meta.env.VITE_FORECAST_REGION_NAME || DEFAULT_FORECAST_REGION_NAME).trim();
}
