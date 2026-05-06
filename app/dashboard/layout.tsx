/**
 * Dashboard chrome — shared header + sign-out for every authed page.
 */
import Link from "next/link";
import { UserButton } from "@clerk/nextjs";

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const clerkConfigured = Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY);
  return (
    <div className="min-h-screen bg-[color:var(--background)]">
      <header className="border-b border-[color:var(--muted)]/20 px-6 py-3 flex items-center justify-between">
        <Link href="/dashboard" className="font-medium">
          Wildfire Nowcast
        </Link>
        <nav className="flex items-center gap-4 text-sm">
          <Link href="/dashboard/aoi/new" className="underline">
            New AOI
          </Link>
          <Link href="/api/export/aois.geojson" className="underline">
            Export AOIs
          </Link>
          <Link href="/api/export/briefs.csv" className="underline">
            Export briefs
          </Link>
          {clerkConfigured ? <UserButton /> : null}
        </nav>
      </header>
      <div className="mx-auto max-w-5xl px-6 py-8">{children}</div>
    </div>
  );
}
