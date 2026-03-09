import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("../config/runtime", () => ({
  apiBaseUrlCandidates: () => ["http://bad-host", "http://good-host"]
}));

import { ApiError, getDataFreshnessStatus } from "../api/client";

describe("api client fallback behavior", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("falls back to next API candidate when first is unavailable", async () => {
    const fetchMock = vi
      .fn()
      .mockRejectedValueOnce(new TypeError("network failure"))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ overall_state: "healthy", sources: {} }), {
          status: 200,
          headers: { "Content-Type": "application/json" }
        })
      );

    vi.stubGlobal("fetch", fetchMock);

    const data = await getDataFreshnessStatus();

    expect(data.overall_state).toBe("healthy");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("surfaces API error on non-200 response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "Bad request" }), {
        status: 400,
        headers: { "Content-Type": "application/json" }
      })
    );

    vi.stubGlobal("fetch", fetchMock);

    await expect(getDataFreshnessStatus()).rejects.toBeInstanceOf(ApiError);
  });
});
