/**
 * Share-this-brief toggle. Mints / revokes via /api/brief/[id]/share.
 */
"use client";

import { useState } from "react";

export function ShareToggle({
  briefId,
  initialToken,
  initialExpiresAt,
}: {
  briefId: string;
  initialToken: string | null;
  initialExpiresAt: string | null;
}) {
  const [token, setToken] = useState<string | null>(initialToken);
  const [expiresAt, setExpiresAt] = useState<string | null>(initialExpiresAt);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function mint() {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/brief/${briefId}/share`, { method: "POST" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const body = (await res.json()) as { token: string; expiresAt: string };
      setToken(body.token);
      setExpiresAt(body.expiresAt);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Mint failed");
    } finally {
      setBusy(false);
    }
  }

  async function revoke() {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/brief/${briefId}/share`, { method: "DELETE" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setToken(null);
      setExpiresAt(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Revoke failed");
    } finally {
      setBusy(false);
    }
  }

  const url = token ? `/brief/share/${token}` : null;

  return (
    <div className="rounded border p-3 text-sm">
      <p className="font-medium">Public share</p>
      {token ? (
        <>
          <p className="mt-1">
            Public URL:{" "}
            <a className="underline" href={url!} target="_blank" rel="noreferrer">
              {url}
            </a>
          </p>
          <p className="text-xs text-[color:var(--muted)]">
            Expires {expiresAt}
          </p>
          <button
            type="button"
            onClick={revoke}
            disabled={busy}
            className="mt-2 rounded border px-3 py-1 text-xs"
          >
            Revoke link
          </button>
        </>
      ) : (
        <button
          type="button"
          onClick={mint}
          disabled={busy}
          className="mt-2 rounded bg-[color:var(--accent)] px-3 py-1 text-xs text-white"
        >
          Create public link
        </button>
      )}
      {error ? <p className="mt-2 text-red-700">{error}</p> : null}
    </div>
  );
}
