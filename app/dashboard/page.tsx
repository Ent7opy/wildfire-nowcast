/**
 * Dashboard home — list of the user's AOIs with last-brief timestamp.
 */
import Link from "next/link";
import { tryGetDb } from "@/lib/db/client";
import { requireUserId, ensureUserExists } from "@/lib/auth/context";
import { listAoisWithLatestBrief, type AoiListRow } from "@/lib/db/aoi-repository";
import { AoiList } from "./_components/aoi-list";

export default async function DashboardPage() {
  const db = tryGetDb();
  if (!db) {
    return <ConfigBanner reason="DATABASE_URL not configured" />;
  }
  const auth = await requireUserId();
  if (!auth.ok) {
    if (auth.code === "config_missing") {
      return <ConfigBanner reason="Auth not configured" />;
    }
    return <ConfigBanner reason="Sign in required" />;
  }
  await ensureUserExists(db, auth.userId);
  const rows: AoiListRow[] = await listAoisWithLatestBrief(db, auth.userId);

  return <AoiList rows={rows} />;
}

function ConfigBanner({ reason }: { reason: string }) {
  return (
    <div className="rounded-md border border-yellow-300 bg-yellow-50 p-4 text-sm text-yellow-900">
      <p className="font-medium">Dashboard unavailable</p>
      <p className="mt-1">{reason}.</p>
      <p className="mt-2">
        <Link className="underline" href="/">
          Back to home
        </Link>
      </p>
    </div>
  );
}
