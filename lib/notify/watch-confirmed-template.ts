/**
 * Stage 9 — watch-confirmed email body renderer.
 *
 * Produces a deterministic markdown body for the one-shot "now watching ..."
 * email dispatched by `dispatchWatchConfirmed` after AOI creation.
 */
import { humanizeRegionBucket } from "@/lib/geo/region-bucket";

export type WatchConfirmedTemplateArgs = {
  aoiName: string;
  regionBucket: string;
  areaHa: number;
  firstPollAt: Date;
  aoiUrl: string;
};

export type RenderedEmail = {
  subject: string;
  markdown: string;
};

export function renderWatchConfirmedEmail(
  args: WatchConfirmedTemplateArgs,
): RenderedEmail {
  const subject = `Now watching ${args.aoiName}`;
  const region = humanizeRegionBucket(args.regionBucket);
  const area = formatHectares(args.areaHa);
  const firstPoll = formatUtc(args.firstPollAt);

  const markdown = [
    `# Now watching ${args.aoiName}`,
    ``,
    `Hi,`,
    ``,
    `Your area "${args.aoiName}" (${area}, region: ${region}) is now being watched.`,
    ``,
    `We poll NASA FIRMS every 15 minutes for new fire detections.`,
    `Your first poll is scheduled by ${firstPoll} (usually within 15 minutes).`,
    `If a detection inside or near your polygon meets the alert thresholds,`,
    `you will receive a situation brief.`,
    ``,
    `View this AOI: ${args.aoiUrl}`,
    `Edit alert rules: ${args.aoiUrl}/rules`,
    ``,
    `— Wildfire Nowcast`,
    `Free, open, AI-native fire intelligence for stewardship — depth over speed.`,
    ``,
  ].join("\n");

  return { subject, markdown };
}

function formatHectares(areaHa: number): string {
  if (!Number.isFinite(areaHa)) return "unknown area";
  if (areaHa >= 100) return `${Math.round(areaHa).toLocaleString("en-US")} ha`;
  return `${areaHa.toFixed(1)} ha`;
}

function formatUtc(d: Date): string {
  if (!(d instanceof Date) || !Number.isFinite(d.getTime())) {
    return "the next cron tick";
  }
  const iso = d.toISOString();
  // "2026-05-07T14:32:00Z" -> "2026-05-07 14:32 UTC"
  const date = iso.slice(0, 10);
  const time = iso.slice(11, 16);
  return `${date} ${time} UTC`;
}
