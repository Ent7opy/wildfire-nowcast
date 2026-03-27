import { useEffect } from "react";

/**
 * Comma-separated list of origins that are allowed to send postMessage events
 * to this app. Defaults to earth-tools.org (production embedding host).
 */
const ALLOWED_ORIGINS: ReadonlySet<string> = new Set(
  (import.meta.env.VITE_EMBED_ALLOW_ORIGINS ?? "https://earth-tools.org")
    .split(",")
    .map((o: string) => o.trim())
    .filter(Boolean)
);

export interface ParentMessage {
  type: string;
  [key: string]: unknown;
}

/**
 * Posts a message to the parent frame. No-ops when not embedded.
 */
export function sendToParent(message: ParentMessage): void {
  if (window.self === window.top) return;
  try {
    window.parent.postMessage(message, "*");
  } catch {
    // Swallow cross-origin errors — parent may have navigated away.
  }
}

/**
 * When `isEmbedded` is true:
 *  - Sends a `{ type: "ready" }` message to the parent frame on mount.
 *  - Attaches a `message` listener that validates the sender origin before
 *    forwarding the event to `onMessage`.
 */
export function useParentFrame(
  isEmbedded: boolean,
  onMessage?: (msg: ParentMessage) => void
): void {
  useEffect(() => {
    if (!isEmbedded) return;

    sendToParent({ type: "ready" });

    if (!onMessage) return;
    const handler = onMessage;  // type is now (msg: ParentMessage) => void

    function handleMessage(event: MessageEvent): void {
      if (!ALLOWED_ORIGINS.has(event.origin)) return;
      if (
        typeof event.data !== "object" ||
        event.data === null ||
        typeof (event.data as Record<string, unknown>).type !== "string"
      ) {
        return;
      }
      handler(event.data as ParentMessage);
    }

    window.addEventListener("message", handleMessage);
    return () => window.removeEventListener("message", handleMessage);
  }, [isEmbedded, onMessage]);
}
