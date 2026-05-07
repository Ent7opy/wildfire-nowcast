/**
 * Stage 8 — minimal "47 minutes ago" formatter.
 *
 * In-repo to avoid pulling `date-fns` for one helper. Future-tense and
 * sub-second deltas are out of scope (this is for the freshness banner).
 */

export function formatRelative(now: Date, then: Date): string {
  const ms = now.getTime() - then.getTime();
  if (!Number.isFinite(ms)) return "unknown time ago";
  if (ms < 0) return "just now";
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return "less than a minute ago";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  const days = Math.floor(hours / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}
