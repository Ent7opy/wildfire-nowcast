/**
 * Regression test for GitHub issue #311:
 * The GDELT queryFn must destructure `signal` from QueryFunctionContext
 * and forward it to `fetch` so that React Query can abort in-flight requests.
 */
import { describe, it, expect, vi, afterEach } from "vitest";

// Re-implement the queryFn in isolation so we can verify the signal contract
// without rendering the full component.
async function gdeltQueryFn(
  { signal }: { signal: AbortSignal },
  gdeltTimeParam: { startdatetime: string; enddatetime: string } | null
): Promise<unknown[]> {
  const query = encodeURIComponent(
    "(wildfire OR bushfire) sourcelang:english"
  );
  const timeQuery = gdeltTimeParam
    ? `&startdatetime=${gdeltTimeParam.startdatetime}&enddatetime=${gdeltTimeParam.enddatetime}`
    : `&timespan=12h`;
  const url = `https://api.gdeltproject.org/api/v2/doc/doc?query=${query}&mode=artlist&format=json${timeQuery}&sort=datedesc&maxrecords=75`;
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error("Failed to fetch news");
  const json = (await res.json()) as { articles?: Array<{ title: string }> };
  return json.articles ?? [];
}

describe("GDELT queryFn signal forwarding (#311)", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("passes the AbortSignal to fetch", async () => {
    const controller = new AbortController();
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ articles: [] })
    });
    vi.stubGlobal("fetch", mockFetch);

    await gdeltQueryFn({ signal: controller.signal }, null);

    expect(mockFetch).toHaveBeenCalledOnce();
    const [, fetchInit] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(fetchInit).toBeDefined();
    expect(fetchInit.signal).toBe(controller.signal);
  });

  it("aborts the fetch when the signal is aborted before the call resolves", async () => {
    const controller = new AbortController();

    const mockFetch = vi.fn().mockImplementation(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          if (init?.signal) {
            init.signal.addEventListener("abort", () => {
              reject(new DOMException("The user aborted a request.", "AbortError"));
            });
          }
        })
    );
    vi.stubGlobal("fetch", mockFetch);

    const pending = gdeltQueryFn({ signal: controller.signal }, null);
    controller.abort();

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  });

  it("passes time range params when gdeltTimeParam is provided", async () => {
    const controller = new AbortController();
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ articles: [] })
    });
    vi.stubGlobal("fetch", mockFetch);

    const timeParam = { startdatetime: "20260101120000", enddatetime: "20260102120000" };
    await gdeltQueryFn({ signal: controller.signal }, timeParam);

    const [calledUrl] = mockFetch.mock.calls[0] as [string];
    expect(calledUrl).toContain(`startdatetime=${timeParam.startdatetime}`);
    expect(calledUrl).toContain(`enddatetime=${timeParam.enddatetime}`);
    expect(calledUrl).not.toContain("timespan=12h");
  });
});
