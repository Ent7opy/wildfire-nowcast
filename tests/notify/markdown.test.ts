/**
 * Stage 4/6 — base markdown renderer surface beyond the link allow-list.
 *
 * The renderer is reused for the user-facing brief view (Stage 6), so the
 * XSS surface for raw HTML in payload text matters: every text node must
 * be escaped before structural tags are emitted.
 */
import { describe, expect, it } from "vitest";
import { escapeHtml, renderMarkdownToHtml } from "@/lib/notify/markdown";

describe("escapeHtml", () => {
  it("encodes the five HTML-significant characters", () => {
    expect(escapeHtml(`& < > " '`)).toBe("&amp; &lt; &gt; &quot; &#39;");
  });

  it("is a no-op on plain ASCII text", () => {
    expect(escapeHtml("Spring Creek Preserve")).toBe("Spring Creek Preserve");
  });
});

describe("renderMarkdownToHtml — XSS surface", () => {
  it("escapes a raw <script> tag in paragraph body text", () => {
    // Stage 6 reuses this renderer for the dashboard brief view. A model
    // that emits `<script>` in a summary must not produce live markup.
    const out = renderMarkdownToHtml("Summary: <script>alert(1)</script> end.");
    expect(out).not.toContain("<script>");
    expect(out).toContain("&lt;script&gt;alert(1)&lt;/script&gt;");
  });

  it("escapes raw HTML inside a heading", () => {
    const out = renderMarkdownToHtml("# <img src=x onerror=alert(1)>");
    expect(out).toContain("<h1>&lt;img src=x onerror=alert(1)&gt;</h1>");
    expect(out).not.toContain("<img");
  });

  it("escapes raw HTML inside a bullet item", () => {
    const out = renderMarkdownToHtml("- <iframe src=//evil></iframe>");
    expect(out).toContain("<li>&lt;iframe src=//evil&gt;&lt;/iframe&gt;</li>");
    expect(out).not.toContain("<iframe");
  });
});

describe("renderMarkdownToHtml — structure", () => {
  it("renders inline _italic_ inside a paragraph", () => {
    const out = renderMarkdownToHtml("This is _emphasised_ inline.");
    expect(out).toBe("<p>This is <em>emphasised</em> inline.</p>");
  });

  it("doesn't crash on out-of-spec ### h3 input and preserves the text", () => {
    // `lib/ai/render.ts` only emits `#` and `## `, so deeper headings are
    // out-of-spec input. Assert the heading text survives the render without
    // pinning the specific fallthrough markup — a future contributor adding
    // real h3 support shouldn't have to delete this test.
    const out = renderMarkdownToHtml("### Subsection");
    expect(out).toContain("Subsection");
  });
});
