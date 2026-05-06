/**
 * Tiny Markdown→HTML renderer scoped to the brief shape produced by
 * `lib/ai/render.ts`:
 *   - `# heading` and `## heading`
 *   - blank-line paragraphs
 *   - `- bullet` lists
 *   - `_italic line_`
 *   - inline `_..._` italics inside paragraphs
 *
 * Why not `marked`: the brief renderer is deterministic and the surface
 * we need is small. A new dependency for ~50 LOC of escapes + line
 * folding is heavier than the function itself. This converter is
 * exercised by the canonical Spring Creek snapshot test.
 *
 * HTML escaping: every text node is escaped before structural tags are
 * emitted, so user-visible content from the brief payload (AOI name,
 * summary, watch items, etc.) cannot inject markup.
 */

const ESC: Record<string, string> = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ESC[c]);
}

/** Replace `_text_` with `<em>text</em>`. Applied to already-escaped text. */
function inlineItalics(escaped: string): string {
  return escaped.replace(/_([^_\n]+)_/g, "<em>$1</em>");
}

/**
 * Replace `[label](url)` with `<a href="url">label</a>` on already-escaped
 * text. Both label and URL come from text that was HTML-escaped *before*
 * this function ran, so `"` in the URL is already `&quot;` and is safe to
 * place inside the double-quoted href attribute.
 *
 * Nested links and labels containing `]` or `(` are not supported and fall
 * through unchanged (they render as plain escaped text).
 */
function inlineLinks(escaped: string): string {
  // Operate on the escaped string. Brackets/parens are not in the HTML escape
  // set, so they survive verbatim — but the label cannot contain `]` and the
  // URL cannot contain `)`, which keeps the regex unambiguous.
  return escaped.replace(
    /\[([^\]\n]+)\]\(([^)\s]+)\)/g,
    (_match, label: string, url: string) => `<a href="${url}">${label}</a>`,
  );
}

export function renderMarkdownToHtml(md: string): string {
  const lines = md.split(/\r?\n/);
  const out: string[] = [];
  let i = 0;
  let inList = false;

  const closeList = () => {
    if (inList) {
      out.push("</ul>");
      inList = false;
    }
  };

  while (i < lines.length) {
    const raw = lines[i];
    const line = raw.trimEnd();

    if (line === "") {
      closeList();
      i += 1;
      continue;
    }

    if (line.startsWith("# ")) {
      closeList();
      out.push(`<h1>${inlineLinks(inlineItalics(escapeHtml(line.slice(2))))}</h1>`);
      i += 1;
      continue;
    }
    if (line.startsWith("## ")) {
      closeList();
      out.push(`<h2>${inlineLinks(inlineItalics(escapeHtml(line.slice(3))))}</h2>`);
      i += 1;
      continue;
    }
    if (line.startsWith("- ")) {
      if (!inList) {
        out.push("<ul>");
        inList = true;
      }
      out.push(`<li>${inlineLinks(inlineItalics(escapeHtml(line.slice(2))))}</li>`);
      i += 1;
      continue;
    }

    closeList();
    const paraLines: string[] = [line];
    i += 1;
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !lines[i].startsWith("#") &&
      !lines[i].startsWith("- ")
    ) {
      paraLines.push(lines[i].trimEnd());
      i += 1;
    }
    const joined = paraLines.join(" ");
    out.push(`<p>${inlineLinks(inlineItalics(escapeHtml(joined)))}</p>`);
  }

  closeList();
  return out.join("\n");
}
