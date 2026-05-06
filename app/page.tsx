import Link from "next/link";
import { POSITIONING_LINE, REPO_URL } from "@/lib/export/positioning";

export default function Home() {
  const clerkConfigured = Boolean(
    process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY,
  );

  return (
    <main className="mx-auto flex min-h-screen max-w-2xl flex-col justify-center gap-8 px-6 py-16">
      <header className="flex flex-col gap-4">
        <p className="text-sm uppercase tracking-[0.2em] text-[color:var(--muted)]">
          Wildfire Nowcast
        </p>
        <h1 className="text-3xl font-medium leading-tight text-[color:var(--foreground)] sm:text-4xl">
          Free, open, AI-native fire intelligence for stewardship —{" "}
          <span className="text-[color:var(--accent)]">depth over speed</span>.
        </h1>
      </header>

      <section className="flex flex-col gap-4 text-base leading-relaxed text-[color:var(--foreground)]">
        <p>
          Most monitoring tools are built for the fastest possible alert to the
          largest possible audience. That is a real job, and other people are
          doing it well. This is a different job. This tool is built for the
          people whose relationship to land is stewardship — the conservation
          staff, the preserve managers, the collaborative project leads who
          need to know, at 4 a.m. on a Sunday, what is happening on their
          specific acres and whether tonight is a &quot;keep sleeping&quot; or a
          &quot;call Jim&quot; night.
        </p>
        <p>
          You bring the polygons you actually care about. We watch them. When
          something changes, we write you a short brief — one paragraph, in
          plain English, that names its sources, sets the detection in the
          context of your land&apos;s history, and is explicit about what it is
          not recommending. Fire Stewardship Agent runs under Earth Tools,
          non-profit by intent, on free-tier infrastructure, with the code open
          from day one.
        </p>
        <p>
          Depth over speed is the design constraint we are holding ourselves
          to. One small tool, held close to one real problem, for as long as it
          remains genuinely useful.
        </p>
      </section>

      {clerkConfigured ? (
        <div className="rounded-md border border-[color:var(--accent)]/40 bg-[color:var(--accent)]/5 p-4 text-sm">
          <Link
            href="/sign-in"
            className="font-medium text-[color:var(--accent)] underline"
          >
            Sign in to start watching
          </Link>
          .
        </div>
      ) : (
        <footer className="rounded-md border border-[color:var(--muted)]/20 bg-[color:var(--foreground)]/5 p-4 text-sm text-[color:var(--muted)]">
          <p>
            <span className="font-medium text-[color:var(--foreground)]">
              Not ready yet.
            </span>{" "}
            This is a placeholder for the A&rsquo; pivot. Fire Stewardship Agent
            v1 targets Q2 2026 and will launch first inside the Land Trust
            Alliance Wildfire Resilience Network. No sign-up yet; nothing to
            click.
          </p>
        </footer>
      )}

      <footer className="border-t border-[color:var(--muted)]/20 pt-4 text-xs text-[color:var(--muted)]">
        <p>{POSITIONING_LINE}</p>
        <p className="mt-1">
          <a className="underline" href={REPO_URL} target="_blank" rel="noreferrer">
            View source
          </a>
        </p>
      </footer>
    </main>
  );
}
