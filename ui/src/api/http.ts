import { apiBaseUrlCandidates } from "../config/runtime";
export { isoFormat } from "../utils/time";

export class ApiError extends Error {
  statusCode?: number;
  url?: string;
  responseText?: string;

  constructor(message: string, options?: { statusCode?: number; url?: string; responseText?: string }) {
    super(message);
    this.name = "ApiError";
    this.statusCode = options?.statusCode;
    this.url = options?.url;
    this.responseText = options?.responseText;
  }
}

export class ApiUnavailableError extends ApiError {
  constructor(message: string, options?: { url?: string }) {
    super(message, options);
    this.name = "ApiUnavailableError";
  }
}

export const GET_CONNECT_TIMEOUT = 2_000;
export const GET_READ_TIMEOUT = 8_000;
export const GET_RETRY_READ_TIMEOUT = 15_000;


export function withTimeout(timeoutMs: number): { signal: AbortSignal; cancel: () => void } {
  const controller = new AbortController();
  const id = globalThis.setTimeout(() => controller.abort(), timeoutMs);
  return {
    signal: controller.signal,
    cancel: () => globalThis.clearTimeout(id)
  };
}

export function toSearchParams(params: Record<string, unknown>): string {
  const searchParams = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value === null || value === undefined) {
      return;
    }
    if (typeof value === "boolean") {
      searchParams.set(key, value ? "true" : "false");
      return;
    }
    searchParams.set(key, String(value));
  });
  return searchParams.toString();
}

export async function getJson<T>(
  path: string,
  params: Record<string, unknown>,
  options?: { slowPath?: boolean }
): Promise<T> {
  const candidates = apiBaseUrlCandidates();
  let lastUnavailable: ApiUnavailableError | null = null;

  for (const base of candidates) {
    const query = toSearchParams(params);
    const url = `${base}${path}${query ? `?${query}` : ""}`;

    const firstAttempt = withTimeout(GET_CONNECT_TIMEOUT + GET_READ_TIMEOUT);
    try {
      const response = await fetch(url, { method: "GET", signal: firstAttempt.signal });
      firstAttempt.cancel();
      if (!response.ok) {
        const text = await response.text();
        throw new ApiError("Non-200 response from API", {
          statusCode: response.status,
          url,
          responseText: text
        });
      }
      return (await response.json()) as T;
    } catch (error) {
      firstAttempt.cancel();
      const aborted = error instanceof DOMException && error.name === "AbortError";
      const shouldRetry = options?.slowPath && aborted;

      if (shouldRetry) {
        const secondAttempt = withTimeout(GET_CONNECT_TIMEOUT + GET_RETRY_READ_TIMEOUT);
        try {
          const response = await fetch(url, { method: "GET", signal: secondAttempt.signal });
          secondAttempt.cancel();
          if (!response.ok) {
            const text = await response.text();
            throw new ApiError("Non-200 response from API", {
              statusCode: response.status,
              url,
              responseText: text
            });
          }
          return (await response.json()) as T;
        } catch (innerErr) {
          secondAttempt.cancel();
          const message = innerErr instanceof Error ? innerErr.message : "API unavailable";
          lastUnavailable = new ApiUnavailableError(message, { url });
          continue;
        }
      }

      if (error instanceof ApiError) {
        throw error;
      }

      const message = error instanceof Error ? error.message : "API unavailable";
      lastUnavailable = new ApiUnavailableError(message, { url });
    }
  }

  if (lastUnavailable) {
    throw lastUnavailable;
  }

  throw new ApiUnavailableError("API unavailable");
}

export async function postJson<T>(path: string, payload: Record<string, unknown>, acceptedStatus = 200, method: "POST" | "PUT" = "POST"): Promise<T> {
  const candidates = apiBaseUrlCandidates();
  let lastUnavailable: ApiUnavailableError | null = null;

  for (const base of candidates) {
    const url = `${base}${path}`;
    const timeout = withTimeout(15_000);
    try {
      const response = await fetch(url, {
        method,
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload),
        signal: timeout.signal
      });
      timeout.cancel();

      if (response.status !== acceptedStatus) {
        const text = await response.text();
        let message = `Non-${acceptedStatus} response from API`;
        try {
          const parsed = JSON.parse(text) as { detail?: string; message?: string };
          message = parsed.message || parsed.detail || message;
        } catch {
          // ignore parse errors
        }
        throw new ApiError(message, { statusCode: response.status, url, responseText: text });
      }

      return (await response.json()) as T;
    } catch (error) {
      timeout.cancel();
      if (error instanceof ApiError) {
        throw error;
      }
      const message = error instanceof Error ? error.message : "API unavailable";
      lastUnavailable = new ApiUnavailableError(message, { url });
    }
  }

  if (lastUnavailable) {
    throw lastUnavailable;
  }

  throw new ApiUnavailableError("API unavailable");
}

export function putJson<T>(path: string, payload: Record<string, unknown>): Promise<T> {
  return postJson<T>(path, payload, 200, "PUT");
}

export async function deleteRequest(path: string): Promise<void> {
  const candidates = apiBaseUrlCandidates();
  let lastUnavailable: ApiUnavailableError | null = null;

  for (const base of candidates) {
    const url = `${base}${path}`;
    const timeout = withTimeout(10_000);
    try {
      const response = await fetch(url, { method: "DELETE", signal: timeout.signal });
      timeout.cancel();
      if (!response.ok && response.status !== 204) {
        const text = await response.text();
        throw new ApiError("Non-204 response from API", { statusCode: response.status, url, responseText: text });
      }
      return;
    } catch (error) {
      timeout.cancel();
      if (error instanceof ApiError) throw error;
      const message = error instanceof Error ? error.message : "API unavailable";
      lastUnavailable = new ApiUnavailableError(message, { url });
    }
  }

  if (lastUnavailable) throw lastUnavailable;
  throw new ApiUnavailableError("API unavailable");
}
