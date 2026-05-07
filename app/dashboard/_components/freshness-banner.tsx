/**
 * Stage 8 — per-AOI freshness banner.
 *
 * Reads the most recent completed `job_runs` row for the AOI's bucket via
 * `getAoiFreshness` and renders one of:
 *   - happy: "Last polled 47 minutes ago"
 *   - degraded (yellow): rate-limited / network error / timeout / partial /
 *     stale-success
 *   - first-poll-pending: when no completed run exists yet
 *
 * Honest-about-its-own-honesty: never silently renders nothing.
 *
 * Server component. `__testNow` is a test-only inject; the page-level caller
 * never passes it. Pure server-side; no client JS.
 */
import type { AppDb } from "@/lib/db/client";
import { getAoiFreshness } from "@/lib/db/freshness";
import { formatRelative } from "@/lib/ui/relative-time";

type Props = {
  db: AppDb;
  aoiId: string;
  userId: string;
  __testNow?: Date;
};

export async function FreshnessBanner({ db, aoiId, userId, __testNow }: Props) {
  const now = __testNow ?? new Date();
  const freshness = await getAoiFreshness(db, { aoiId, userId, now });

  if (!freshness || !freshness.lastPolledAt) {
    return (
      <BannerShell tone="warn">
        First poll pending — usually within 15 minutes
      </BannerShell>
    );
  }

  const rel = formatRelative(now, freshness.lastPolledAt);

  if (freshness.outcome === "rate_limited") {
    return (
      <BannerShell tone="warn">
        Last attempt: rate-limited{freshness.retryPending ? " — retrying next tick" : ""} ({rel})
      </BannerShell>
    );
  }
  if (freshness.outcome === "network_error") {
    return (
      <BannerShell tone="warn">
        Last attempt: network error{freshness.retryPending ? " — retrying next tick" : ""} ({rel})
      </BannerShell>
    );
  }
  if (freshness.outcome === "timeout") {
    return (
      <BannerShell tone="warn">
        Last attempt: timed out{freshness.retryPending ? " — retrying next tick" : ""} ({rel})
      </BannerShell>
    );
  }
  if (freshness.outcome === "partial") {
    return (
      <BannerShell tone="warn">
        Last attempt: partial — some AOIs failed ({rel})
      </BannerShell>
    );
  }
  if (freshness.isStale) {
    return (
      <BannerShell tone="warn">
        Polling delayed — last successful tick over 30 minutes ago ({rel})
      </BannerShell>
    );
  }
  return <BannerShell tone="muted">Last polled {rel}</BannerShell>;
}

function BannerShell({
  tone,
  children,
}: {
  tone: "muted" | "warn";
  children: React.ReactNode;
}) {
  const className =
    tone === "warn"
      ? "rounded border border-[color:var(--warn)] bg-[color:var(--warn)]/10 px-3 py-2 text-sm text-[color:var(--foreground)]"
      : "text-xs text-[color:var(--muted)]";
  return <div className={className}>{children}</div>;
}
