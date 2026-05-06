# Brief 08 — Adversarial critic: candidate D ("FIRMS done right" substrate)

## Why this exists

Candidate D ("FIRMS done right" as open-source library + normalized substrate) is the recommended *layer* under A and E. Your job is to kill it — prove it's pointless, duplicative, or un-maintainable for a solo operator.

**Read first:**
1. `pm/PM_CLAUDE.md`
2. `pm/decisions/0002-phase-1-synthesis.md`
3. `pm/decisions/0003-nonprofit-and-free-infra-constraints.md`
4. `pm/backlog.md` § "D — FIRMS done right"
5. `pm/research-log/2026-04-21-github.md` — the pro-argument is here

## Goal

The strongest possible case that candidate D is not worth building. Force PM to justify it against concrete alternatives.

## Attack vectors to press on

1. **NASA could upgrade FIRMS itself.** NASA has an incentive to fix endpoint drift, dedup, and lineage in their own API. If they do (check roadmap, announcements, recent API versions), the library is obsolete. Real risk?
2. **Duplicative with existing wrappers.** Is there actually an OSS wrapper that's good enough — just under-advertised? Dig past `datadesk/nasa-wildfires` (34 stars, thin). Check `pyrosm`, GDAL/OGR connectors, STAC catalogs, scientific-Python packages for FIRMS. Is the "nobody has built this" claim robust?
3. **Audience is too small to matter.** Count actual OSS FIRMS-consumer projects and their activity. If the realistic audience is <100 devs globally, "another library" is not a product — it's a weekend donation.
4. **Industrial masking specifics.** The WFN industrial-coverage work is described as unusual. But is the industrial dataset itself (OSM industrial polygons + NASA thermal anomaly mask) actually curate-once-and-done, meaning the perceived moat decays fast?
5. **Maintenance cost of an OSS library solo.** What's the historical failure rate of solo-maintained data-infrastructure libraries? What's the support burden when a random dev opens an issue at 2am?
6. **Doesn't directly produce a product.** A library is infrastructure. Infrastructure alone doesn't help a non-profit fundraise or help Vanyo's Accedia talk tell a coherent product story. Is this diffuse-impact?
7. **Competes with Vanyo's time on candidate A.** If we only do A, and use a thin internal FIRMS layer without publishing it, is that strictly worse than A + public-D?

## Constraints

Same as brief 07. Cite evidence. Name sources. Distinguish fatal from manageable.

## Output

**`pm/research-log/2026-04-21-critique-d.md`** — ≤900 words:
- `## Thesis being attacked`
- `## Existing alternatives actually surveyed` — list each OSS FIRMS wrapper / STAC catalog / scientific lib, with stars, activity, and verdict on whether it covers D's space
- `## Strongest attacks` — top 5
- `## NASA roadmap signal` — what's in FIRMS release notes / EarthData announcements 2024–2026
- `## Objections that didn't hold`
- `## Net verdict` — kill / fold-into-A-only / publish-as-byproduct / proceed

**`pm/signals/2026-04-21-critique-d-raw.md`** — citations.

## Time budget

~30 min.
