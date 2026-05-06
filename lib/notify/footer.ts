/**
 * Stage 7 — email footer with the four signed-token action links.
 *
 * Appended verbatim to the brief Markdown by the dispatcher just before
 * `sendEmail`. Order is stable so smoke tests can string-match.
 */

export type FooterUrls = {
  feedbackYesUrl: string;
  feedbackNoUrl: string;
  snoozeUrl: string;
  pauseUrl: string;
  unsubscribeUrl: string;
};

export function renderFooterMarkdown(urls: FooterUrls): string {
  return [
    "",
    "---",
    `Was this brief useful? [Yes](${urls.feedbackYesUrl}) · [No](${urls.feedbackNoUrl})`,
    `· [Snooze 24h](${urls.snoozeUrl}) · [Pause this AOI](${urls.pauseUrl}) · [Unsubscribe](${urls.unsubscribeUrl})`,
    "",
  ].join("\n");
}

export function appendFooter(markdown: string, urls: FooterUrls): string {
  const trimmed = markdown.replace(/\s+$/, "");
  return `${trimmed}\n${renderFooterMarkdown(urls)}`;
}
