/**
 * Public URL helper for shared brief links. Reads the canonical host from
 * `NEXT_PUBLIC_APP_URL` (set by the platform) and falls back to a relative
 * path so the value is still meaningful in dev / build-without-blocking.
 */
export function publicShareUrl(token: string): string {
  const host = process.env.NEXT_PUBLIC_APP_URL?.replace(/\/$/, "") ?? "";
  return `${host}/brief/share/${token}`;
}
