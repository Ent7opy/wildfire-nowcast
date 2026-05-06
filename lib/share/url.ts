/**
 * Public URL helper for shared brief links. Reads the canonical host from
 * `NEXT_PUBLIC_APP_URL` (set by the platform) and falls back to a relative
 * path so the value is still meaningful in dev / build-without-blocking.
 */
export function publicShareUrl(token: string): string {
  const host = process.env.NEXT_PUBLIC_APP_URL?.replace(/\/$/, "") ?? "";
  return `${host}/brief/share/${token}`;
}

/**
 * Stage 7 — public URLs for the four notify-action endpoints. Same
 * `NEXT_PUBLIC_APP_URL` host pattern as `publicShareUrl`.
 */
export function notifyActionUrl(
  action: "snooze" | "pause" | "unsubscribe" | "feedback",
  token: string,
  query?: Record<string, string>,
): string {
  const host = process.env.NEXT_PUBLIC_APP_URL?.replace(/\/$/, "") ?? "";
  const base = `${host}/api/notify/${action}/${token}`;
  if (!query || Object.keys(query).length === 0) return base;
  const qs = new URLSearchParams(query).toString();
  return `${base}?${qs}`;
}
