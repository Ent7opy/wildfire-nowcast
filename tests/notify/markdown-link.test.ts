/**
 * Stage 7 — markdown link extension.
 */
import { describe, expect, it } from "vitest";
import { renderMarkdownToHtml } from "@/lib/notify/markdown";

describe("markdown link rendering", () => {
  it("renders [label](url) as an anchor", () => {
    const out = renderMarkdownToHtml("Click [here](https://example.org/x) now.");
    expect(out).toContain('<a href="https://example.org/x">here</a>');
  });

  it("escapes a quote in the URL (escape happens before link rewrite)", () => {
    const out = renderMarkdownToHtml(`See [link](https://x/q?a=1"&b=2) end.`);
    // The original `"` in URL is escaped to &quot; before link substitution,
    // so the href contains the encoded entity rather than a raw quote.
    expect(out).toContain("&quot;");
    expect(out).not.toContain('q?a=1"&b=2"');
  });

  it("escapes brackets/parens in label content via HTML escape (no XSS)", () => {
    const out = renderMarkdownToHtml("Inline [<script>](https://x) attempt.");
    expect(out).not.toContain("<script>");
    expect(out).toContain("&lt;script&gt;");
  });

  it("plain text without a link is unchanged", () => {
    const out = renderMarkdownToHtml("Just a sentence.");
    expect(out).toBe("<p>Just a sentence.</p>");
  });

  it("works inside a heading and bullet list", () => {
    const md = "## Links\n\n- [A](https://a)\n- [B](https://b)";
    const out = renderMarkdownToHtml(md);
    expect(out).toContain("<h2>Links</h2>");
    expect(out).toContain('<a href="https://a">A</a>');
    expect(out).toContain('<a href="https://b">B</a>');
  });
});
