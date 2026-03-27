// Computed once at module load — the page cannot dynamically move in or out of
// an iframe, so this value is constant for the entire session.
const _isEmbedded = (() => {
  const params = new URLSearchParams(window.location.search);
  if (params.get("embed") === "true") return true;
  try {
    return window.self !== window.top;
  } catch {
    // Cross-origin access throws — we are definitely inside an iframe.
    return true;
  }
})();

/**
 * Returns true when the app is running inside an iframe or when the
 * `?embed=true` query parameter is present (useful for local testing).
 */
export function useEmbedMode(): boolean {
  return _isEmbedded;
}
