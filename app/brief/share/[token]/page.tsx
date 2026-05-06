/**
 * Public, unauthenticated brief view consumed via share-token.
 *
 * Reads via `getBriefByShareToken` which already enforces expiry + non-archived
 * AOI; misses → 404.
 */
import { notFound } from "next/navigation";
import { tryGetDb } from "@/lib/db/client";
import { getBriefByShareToken } from "@/lib/db/aoi-repository";
import { renderMarkdownToHtml } from "@/lib/notify/markdown";
import { POSITIONING_LINE, REPO_URL } from "@/lib/export/positioning";

type Params = { params: Promise<{ token: string }> };

export default async function PublicSharePage({ params }: Params) {
  const { token } = await params;
  const db = tryGetDb();
  if (!db) notFound();
  const brief = await getBriefByShareToken(db, token);
  if (!brief) notFound();

  const html = renderMarkdownToHtml(brief.renderedMarkdown);

  return (
    <main className="mx-auto max-w-3xl px-6 py-10">
      <article
        className="prose prose-sm max-w-none"
        dangerouslySetInnerHTML={{ __html: html }}
      />
      <footer className="mt-10 border-t pt-4 text-xs text-gray-500">
        <p>
          Model: <code>{brief.model}</code> · prompt {brief.promptVersion} ·
          posted {brief.createdAt.toISOString()}
        </p>
        <p className="mt-2">
          {POSITIONING_LINE}{" "}
          <a className="underline" href={REPO_URL} target="_blank" rel="noreferrer">
            View source
          </a>
          .
        </p>
      </footer>
    </main>
  );
}
