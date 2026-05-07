/**
 * Renders the user's AOI list. Pure-data; tested in isolation by feeding
 * fixture rows in.
 */
import Link from "next/link";
import type { AoiListRow } from "@/lib/db/aoi-repository";

export function AoiList({ rows }: { rows: AoiListRow[] }) {
  if (rows.length === 0) {
    return (
      <div className="rounded-md border border-[color:var(--muted)]/30 p-6">
        <h2 className="text-lg font-medium">No AOIs yet</h2>
        <p className="mt-1 text-sm text-[color:var(--muted)]">
          Define a polygon you care about. We&rsquo;ll watch it.
        </p>
        <Link
          href="/dashboard/aoi/new"
          className="mt-4 inline-block rounded bg-[color:var(--accent)] px-4 py-2 text-sm font-medium text-white"
        >
          Create your first AOI
        </Link>
      </div>
    );
  }
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="border-b text-left">
            <th scope="col" className="py-2 pr-4">Name</th>
            <th scope="col" className="py-2 pr-4">Area (ha)</th>
            <th scope="col" className="py-2 pr-4">Region</th>
            <th scope="col" className="py-2 pr-4">Created</th>
            <th scope="col" className="py-2 pr-4">Last brief</th>
            <th scope="col" className="py-2 pr-4">Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className="border-b">
              <td className="py-2 pr-4">
                <Link href={`/dashboard/aoi/${r.id}`} className="underline">
                  {r.name}
                </Link>
              </td>
              <td className="py-2 pr-4">{r.areaHa.toFixed(1)}</td>
              <td className="py-2 pr-4">{r.regionBucket}</td>
              <td className="py-2 pr-4">{r.createdAt.toISOString().slice(0, 10)}</td>
              <td className="py-2 pr-4">
                {r.lastBriefAt ? r.lastBriefAt.toISOString().slice(0, 10) : "—"}
              </td>
              <td className="py-2 pr-4">
                {r.pausedUntil && r.pausedUntil > new Date() ? (
                  <span className="text-yellow-700">paused</span>
                ) : (
                  <span className="text-green-700">active</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
