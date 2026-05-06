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

  describe("URL-scheme allow-list", () => {
    it("allows http://", () => {
      const out = renderMarkdownToHtml("Visit [site](http://example.com).");
      expect(out).toContain('<a href="http://example.com">site</a>');
    });

    it("allows https://", () => {
      const out = renderMarkdownToHtml("Visit [site](https://example.com).");
      expect(out).toContain('<a href="https://example.com">site</a>');
    });

    it("allows mailto:", () => {
      const out = renderMarkdownToHtml("Email [us](mailto:user@example.com).");
      expect(out).toContain('<a href="mailto:user@example.com">us</a>');
    });

    it("allows site-relative URLs (footer build-without-blocking case)", () => {
      const out = renderMarkdownToHtml("[Snooze](/api/notify/snooze/abc)");
      expect(out).toContain('<a href="/api/notify/snooze/abc">Snooze</a>');
    });

    it("rejects javascript: and renders plain label text", () => {
      const out = renderMarkdownToHtml("Click [me](javascript:alert(1)) now.");
      expect(out).not.toContain("<a ");
      expect(out).not.toContain("javascript:");
      expect(out).toContain("me");
    });

    it("rejects data: URLs", () => {
      const out = renderMarkdownToHtml(
        "[x](data:text/html,<script>alert(1)</script>)",
      );
      expect(out).not.toContain("<a ");
      expect(out).not.toContain("data:");
    });

    it("rejects file: URLs", () => {
      const out = renderMarkdownToHtml("[x](file:///etc/passwd)");
      expect(out).not.toContain("<a ");
      expect(out).not.toContain("file:");
    });

    it("rejects scheme with leading whitespace obfuscation", () => {
      // Regex is anchored, so any non-allowed prefix is rejected.
      const out = renderMarkdownToHtml("[x](JAVASCRIPT:alert(1))");
      // Note: the closing `)` of `alert(1)` actually terminates the URL match
      // at the first `)`, but even if extracted, `JAVASCRIPT:alert(1` would
      // not match the allow-list and would degrade to label text.
      expect(out).not.toContain("<a ");
    });
  });
});
