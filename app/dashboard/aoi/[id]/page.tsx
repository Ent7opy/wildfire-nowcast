/**
 * Per-AOI editor — summary, rules form, recent briefs.
 */
import Link from "next/link";
import { notFound } from "next/navigation";
import { tryGetDb } from "@/lib/db/client";
import { requireUserId, ensureUserExists } from "@/lib/auth/context";
import {
  getAoiById,
  getRulesByAoiId,
  listBriefsForAoi,
  listMatchedDetectionsForAoi,
} from "@/lib/db/aoi-repository";
import { RulesForm } from "../../_components/rules-form";
import { AoiMapClient } from "../../_components/aoi-map-client";
import { FreshnessBanner } from "../../_components/freshness-banner";

type Params = { params: Promise<{ id: string }> };

export default async function AoiPage({ params }: Params) {
  const { id } = await params;
  const db = tryGetDb();
  if (!db) notFound();
  const auth = await requireUserId();
  if (!auth.ok) notFound();
  await ensureUserExists(db, auth.userId);

  const aoi = await getAoiById(db, auth.userId, id);
  if (!aoi) notFound();
  const rules = await getRulesByAoiId(db, id);
  const briefs = await listBriefsForAoi(db, {
    userId: auth.userId,
    aoiId: id,
    limit: 20,
  });
  const detections = await listMatchedDetectionsForAoi(db, {
    userId: auth.userId,
    aoiId: id,
    sinceDays: 90,
  });

  const [lon, lat] = aoi.centroid.coordinates;
  const bbox = aoi.bbox.coordinates[0];

  return (
    <div className="flex flex-col gap-8">
      <section>
        <h1 className="text-2xl font-medium">{aoi.name}</h1>
        <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1 text-sm sm:grid-cols-4">
          <dt className="text-[color:var(--muted)]">Area</dt>
          <dd>{aoi.areaHa.toFixed(1)} ha</dd>
          <dt className="text-[color:var(--muted)]">Region</dt>
          <dd>{aoi.regionBucket}</dd>
          <dt className="text-[color:var(--muted)]">Centroid</dt>
          <dd>
            {lat.toFixed(3)}, {lon.toFixed(3)}
          </dd>
          <dt className="text-[color:var(--muted)]">Created</dt>
          <dd>{aoi.createdAt.toISOString().slice(0, 10)}</dd>
        </dl>
        <div className="mt-3 text-xs text-[color:var(--muted)]">
          BBox: [{bbox[0][0].toFixed(2)}, {bbox[0][1].toFixed(2)}] →{" "}
          [{bbox[2][0].toFixed(2)}, {bbox[2][1].toFixed(2)}]
        </div>
        <div className="mt-3">
          <FreshnessBanner db={db} aoiId={aoi.id} userId={auth.userId} />
        </div>
        <div className="mt-4">
          <AoiMapClient
            mode="view"
            name={aoi.name}
            polygon={aoi.polygon}
            bbox={aoi.bbox}
            centroid={aoi.centroid}
            detections={detections.map((d) => ({
              lat: d.lat,
              lon: d.lon,
              frpMw: d.frpMw,
              detectedAt: d.detectedAt.toISOString(),
              satellite: d.satellite,
            }))}
          />
          <p className="mt-1 text-xs text-[color:var(--muted)]">
            {detections.length} matched FIRMS detection{detections.length === 1 ? "" : "s"} in the last 90 days.
          </p>
        </div>
        <div className="mt-3 flex gap-3 text-sm">
          <Link className="underline" href={`/api/aoi/${aoi.id}/export?format=geojson`}>
            Download GeoJSON
          </Link>
          <Link className="underline" href={`/api/aoi/${aoi.id}/export?format=markdown`}>
            Download Markdown
          </Link>
        </div>
      </section>

      <section>
        <h2 className="text-lg font-medium">Rules</h2>
        <RulesForm
          aoiId={aoi.id}
          initial={
            rules
              ? {
                  distanceBufferKm: rules.distanceBufferKm,
                  minConfidence: rules.minConfidence,
                  minFrpMw: rules.minFrpMw,
                  quietHours: rules.quietHours,
                  pausedUntil: rules.pausedUntil?.toISOString() ?? null,
                  notifyChannels: rules.notifyChannels,
                }
              : null
          }
        />
      </section>

      <section>
        <h2 className="text-lg font-medium">Recent briefs</h2>
        {briefs.length === 0 ? (
          <p className="text-sm text-[color:var(--muted)]">
            No briefs yet. We&rsquo;ll generate one when something changes.
          </p>
        ) : (
          <ul className="mt-2 flex flex-col gap-1 text-sm">
            {briefs.map((b) => (
              <li key={b.id}>
                <Link className="underline" href={`/dashboard/brief/${b.id}`}>
                  {b.createdAt.toISOString()} — {b.gateReason}
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
