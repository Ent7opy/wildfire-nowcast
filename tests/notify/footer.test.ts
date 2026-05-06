/**
 * Stage 7 — email footer rendering.
 */
import { describe, expect, it } from "vitest";
import { appendFooter, renderFooterMarkdown } from "@/lib/notify/footer";

const URLS = {
  feedbackYesUrl: "https://app/api/notify/feedback/aaa?v=yes",
  feedbackNoUrl: "https://app/api/notify/feedback/aaa?v=no",
  snoozeUrl: "https://app/api/notify/snooze/bbb",
  pauseUrl: "https://app/api/notify/pause/ccc",
  unsubscribeUrl: "https://app/api/notify/unsubscribe/ddd",
};

describe("notify footer", () => {
  it("contains all four URLs in stable order", () => {
    const md = renderFooterMarkdown(URLS);
    expect(md).toContain(URLS.feedbackYesUrl);
    expect(md).toContain(URLS.feedbackNoUrl);
    expect(md).toContain(URLS.snoozeUrl);
    expect(md).toContain(URLS.pauseUrl);
    expect(md).toContain(URLS.unsubscribeUrl);
    const yesAt = md.indexOf(URLS.feedbackYesUrl);
    const snoozeAt = md.indexOf(URLS.snoozeUrl);
    const pauseAt = md.indexOf(URLS.pauseUrl);
    const unsubAt = md.indexOf(URLS.unsubscribeUrl);
    expect(yesAt).toBeLessThan(snoozeAt);
    expect(snoozeAt).toBeLessThan(pauseAt);
    expect(pauseAt).toBeLessThan(unsubAt);
  });

  it("appendFooter joins with a horizontal rule and trims trailing whitespace", () => {
    const out = appendFooter("# Brief\n\nSummary line\n\n", URLS);
    expect(out).toContain("# Brief");
    expect(out).toContain("---");
    expect(out).toContain("[Snooze 24h](https://app/api/notify/snooze/bbb)");
  });

  it("is deterministic across calls", () => {
    expect(renderFooterMarkdown(URLS)).toBe(renderFooterMarkdown(URLS));
  });
});
