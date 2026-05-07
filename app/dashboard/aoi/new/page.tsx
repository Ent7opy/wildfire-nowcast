/**
 * AOI creation page — upload + paste GeoJSON. No map drawing (v1.1).
 */
"use client";

import { useRouter } from "next/navigation";
import dynamic from "next/dynamic";
import { useRef, useState } from "react";

const AoiMap = dynamic(
  () => import("../../_components/aoi-map").then((m) => m.AoiMap),
  { ssr: false, loading: () => <div className="h-[420px] w-full rounded border" /> },
);

type Tab = "upload" | "paste" | "draw";

const TAB_ORDER: Tab[] = ["paste", "upload", "draw"];

export default function NewAoiPage() {
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("paste");
  const tabRefs = useRef<Record<Tab, HTMLButtonElement | null>>({
    paste: null,
    upload: null,
    draw: null,
  });

  function onTabKeyDown(e: React.KeyboardEvent<HTMLDivElement>) {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const idx = TAB_ORDER.indexOf(tab);
    const next =
      e.key === "ArrowRight"
        ? TAB_ORDER[(idx + 1) % TAB_ORDER.length]
        : TAB_ORDER[(idx - 1 + TAB_ORDER.length) % TAB_ORDER.length];
    setTab(next);
    tabRefs.current[next]?.focus();
  }
  const [name, setName] = useState("");
  const [json, setJson] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    let geometry: unknown;
    try {
      geometry = extractGeometry(JSON.parse(json));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid GeoJSON");
      return;
    }
    await postAoi(geometry);
  }

  async function postAoi(geometry: unknown) {
    setBusy(true);
    try {
      const res = await fetch("/api/aoi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, geometry }),
      });
      if (!res.ok) {
        const body = (await res.json().catch(() => ({}))) as {
          error?: { message?: string };
        };
        throw new Error(body.error?.message ?? `HTTP ${res.status}`);
      }
      const body = (await res.json()) as { aoi: { id: string } };
      router.push(`/dashboard/aoi/${body.aoi.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Create failed");
    } finally {
      setBusy(false);
    }
  }

  async function submitDrawn(polygon: { type: "Polygon"; coordinates: number[][][] }) {
    setError(null);
    if (!name.trim()) {
      setError("Name is required before saving the drawn polygon");
      return;
    }
    await postAoi(polygon);
  }

  async function onFile(file: File) {
    const text = await file.text();
    setJson(text);
  }

  return (
    <div className="max-w-2xl">
      <h1 className="text-xl font-medium">Create AOI</h1>
      <div
        role="tablist"
        aria-label="AOI input method"
        onKeyDown={onTabKeyDown}
        className="mt-4 flex gap-2 text-sm"
      >
        <button
          type="button"
          role="tab"
          id="tab-paste"
          aria-controls="panel-paste"
          aria-selected={tab === "paste"}
          tabIndex={tab === "paste" ? 0 : -1}
          ref={(el) => {
            tabRefs.current.paste = el;
          }}
          onClick={() => setTab("paste")}
          className={`rounded px-3 py-1 ${tab === "paste" ? "bg-[color:var(--accent)] text-white" : "border"}`}
        >
          Paste GeoJSON
        </button>
        <button
          type="button"
          role="tab"
          id="tab-upload"
          aria-controls="panel-upload"
          aria-selected={tab === "upload"}
          tabIndex={tab === "upload" ? 0 : -1}
          ref={(el) => {
            tabRefs.current.upload = el;
          }}
          onClick={() => setTab("upload")}
          className={`rounded px-3 py-1 ${tab === "upload" ? "bg-[color:var(--accent)] text-white" : "border"}`}
        >
          Upload .geojson
        </button>
        <button
          type="button"
          role="tab"
          id="tab-draw"
          aria-controls="panel-draw"
          aria-selected={tab === "draw"}
          tabIndex={tab === "draw" ? 0 : -1}
          ref={(el) => {
            tabRefs.current.draw = el;
          }}
          onClick={() => setTab("draw")}
          className={`rounded px-3 py-1 ${tab === "draw" ? "bg-[color:var(--accent)] text-white" : "border"}`}
        >
          Draw on map
        </button>
      </div>

      {tab === "draw" ? (
        <div
          role="tabpanel"
          id="panel-draw"
          aria-labelledby="tab-draw"
          className="mt-6 flex flex-col gap-4"
        >
          <label className="flex flex-col gap-1 text-sm">
            <span>Name</span>
            <input
              type="text"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="rounded border px-2 py-1"
            />
          </label>
          <p className="text-xs text-[color:var(--muted)]">
            Click on the map to add vertices. Double-click to close the polygon
            and save. The drawn polygon is sent to the same /api/aoi endpoint
            used by the Paste and Upload tabs.
          </p>
          <AoiMap mode="draw" onPolygon={(p) => void submitDrawn(p)} />
          {error ? (
            <p className="rounded border border-red-300 bg-red-50 p-2 text-sm text-red-800">
              {error}
            </p>
          ) : null}
          {busy ? <p className="text-sm">Creating…</p> : null}
        </div>
      ) : (
        <form
          role="tabpanel"
          id={tab === "paste" ? "panel-paste" : "panel-upload"}
          aria-labelledby={tab === "paste" ? "tab-paste" : "tab-upload"}
          className="mt-6 flex flex-col gap-4"
          onSubmit={submit}
        >
          <label className="flex flex-col gap-1 text-sm">
            <span>Name</span>
            <input
              type="text"
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="rounded border px-2 py-1"
            />
          </label>

          {tab === "upload" ? (
            <label className="flex flex-col gap-1 text-sm">
              <span>GeoJSON file</span>
              <input
                type="file"
                accept=".geojson,application/geo+json,application/json"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void onFile(f);
                }}
              />
            </label>
          ) : null}

          <label className="flex flex-col gap-1 text-sm">
            <span>GeoJSON</span>
            <textarea
              required
              rows={10}
              value={json}
              onChange={(e) => setJson(e.target.value)}
              className="rounded border p-2 font-mono text-xs"
              placeholder='{"type":"Polygon","coordinates":[[[...]]]}'
            />
          </label>

          {error ? (
            <p className="rounded border border-red-300 bg-red-50 p-2 text-sm text-red-800">
              {error}
            </p>
          ) : null}

          <button
            type="submit"
            disabled={busy}
            className="self-start rounded bg-[color:var(--accent)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {busy ? "Creating…" : "Create AOI"}
          </button>
        </form>
      )}
    </div>
  );
}

function extractGeometry(parsed: unknown): unknown {
  if (!parsed || typeof parsed !== "object") {
    throw new Error("Expected a GeoJSON object");
  }
  const obj = parsed as { type?: string; geometry?: unknown; features?: unknown[] };
  if (obj.type === "Polygon" || obj.type === "MultiPolygon") {
    return obj;
  }
  if (obj.type === "Feature" && obj.geometry) {
    return obj.geometry;
  }
  if (obj.type === "FeatureCollection" && Array.isArray(obj.features)) {
    if (obj.features.length !== 1) {
      throw new Error("FeatureCollection must contain exactly one feature");
    }
    const first = obj.features[0] as { geometry?: unknown };
    if (!first.geometry) throw new Error("Feature has no geometry");
    return first.geometry;
  }
  throw new Error("Unsupported GeoJSON shape");
}
