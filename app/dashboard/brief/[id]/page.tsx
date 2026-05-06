/**
 * Authenticated brief view — full provenance footer + share toggle.
 */
import { notFound } from "next/navigation";
import { tryGetDb } from "@/lib/db/client";
import { requireUserId, ensureUserExists } from "@/lib/auth/context";
import { getBriefByIdForUser } from "@/lib/db/aoi-repository";
import { renderMarkdownToHtml } from "@/lib/notify/markdown";
import { ShareToggle } from "../../_components/share-toggle";

type Params = { params: Promise<{ id: string }> };

export default async function BriefPage({ params }: Params) {
  const { id } = await params;
  const db = tryGetDb();
  if (!db) notFound();
  const auth = await requireUserId();
  if (!auth.ok) notFound();
  await ensureUserExists(db, auth.userId);

  const brief = await getBriefByIdForUser(db, {
    userId: auth.userId,
    briefId: id,
  });
  if (!brief) notFound();

  const html = renderMarkdownToHtml(brief.renderedMarkdown);

  return (
    <article className="flex flex-col gap-6">
      <div
        className="prose prose-sm max-w-none"
        dangerouslySetInnerHTML={{ __html: html }}
      />

      <ShareToggle
        briefId={brief.id}
        initialToken={brief.shareToken}
        initialExpiresAt={brief.shareExpiresAt?.toISOString() ?? null}
      />

      <footer className="border-t pt-4 text-xs text-[color:var(--muted)]">
        <p>Model: {brief.model} · prompt {brief.promptVersion}</p>
        <p>
          Gate reason: {brief.gateReason} · latency:{" "}
          {brief.latencyMs ?? "–"} ms · cost-est: ${brief.costUsdEst ?? "–"}
        </p>
        <p>Posted: {brief.createdAt.toISOString()}</p>
      </footer>
    </article>
  );
}
