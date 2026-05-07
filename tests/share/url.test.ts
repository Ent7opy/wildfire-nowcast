import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { notifyActionUrl, publicShareUrl } from "@/lib/share/url";

describe("publicShareUrl", () => {
  const original = process.env.NEXT_PUBLIC_APP_URL;
  afterEach(() => {
    if (original === undefined) delete process.env.NEXT_PUBLIC_APP_URL;
    else process.env.NEXT_PUBLIC_APP_URL = original;
  });

  it("returns a relative path when NEXT_PUBLIC_APP_URL is unset", () => {
    delete process.env.NEXT_PUBLIC_APP_URL;
    expect(publicShareUrl("abc")).toBe("/brief/share/abc");
  });

  it("strips a single trailing slash from the host", () => {
    process.env.NEXT_PUBLIC_APP_URL = "https://example.com/";
    expect(publicShareUrl("abc")).toBe("https://example.com/brief/share/abc");
  });

  it("preserves a host with no trailing slash", () => {
    process.env.NEXT_PUBLIC_APP_URL = "https://example.com";
    expect(publicShareUrl("abc")).toBe("https://example.com/brief/share/abc");
  });
});

describe("notifyActionUrl", () => {
  const original = process.env.NEXT_PUBLIC_APP_URL;
  beforeEach(() => {
    process.env.NEXT_PUBLIC_APP_URL = "https://example.com";
  });
  afterEach(() => {
    if (original === undefined) delete process.env.NEXT_PUBLIC_APP_URL;
    else process.env.NEXT_PUBLIC_APP_URL = original;
  });

  it("builds a bare action URL when no query is provided", () => {
    expect(notifyActionUrl("snooze", "tok123")).toBe(
      "https://example.com/api/notify/snooze/tok123",
    );
  });

  it("treats an empty query object as no query", () => {
    expect(notifyActionUrl("pause", "tok123", {})).toBe(
      "https://example.com/api/notify/pause/tok123",
    );
  });

  it("appends and URL-encodes query parameters", () => {
    const url = notifyActionUrl("feedback", "tok123", {
      rating: "thumbs up",
      from: "a&b",
    });
    expect(url).toBe(
      "https://example.com/api/notify/feedback/tok123?rating=thumbs+up&from=a%26b",
    );
  });

  it("works for every action variant", () => {
    for (const action of ["snooze", "pause", "unsubscribe", "feedback"] as const) {
      expect(notifyActionUrl(action, "t")).toContain(`/api/notify/${action}/t`);
    }
  });
});
