/**
 * Client wrapper that lets RSC pages mount the dynamically-imported AoiMap
 * with `ssr: false`. Next 16 forbids `dynamic({ ssr: false })` in Server
 * Components, so this small client island owns the dynamic import.
 */
"use client";

import dynamic from "next/dynamic";
import type { AoiMapProps } from "./aoi-map";

const AoiMap = dynamic(() => import("./aoi-map").then((m) => m.AoiMap), {
  ssr: false,
  loading: () => (
    <div className="h-[420px] w-full rounded border border-[color:var(--muted)]" />
  ),
});

export function AoiMapClient(props: AoiMapProps): React.ReactElement {
  return <AoiMap {...props} />;
}
