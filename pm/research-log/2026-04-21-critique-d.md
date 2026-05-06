# Adversarial critique — Candidate D ("FIRMS done right" substrate)

**Agent:** 08
**Stance:** Kill (brief asked for steel-manned attack).
**Verdict:** **Kill as standalone candidate. Fold into A as a ~5-hour byproduct library, not a product.**

## Thesis being attacked

*A clean, open-source, drift-monitored, deduplicated FIRMS-ingest library with industrial false-positive masking is a real adjacency for WFN — fills an OSS gap, serves as substrate under A + E.*

## Existing alternatives surveyed

| Alternative | Stars / status | Covers D? |
|---|---|---|
| **NASA FIRMS API v4.0.66 (first-party)** | mature; `data_availability` + `missing_data` endpoints built-in; Archive Download | Partially — API is canonical; any "done right" lib is a thin layer |
| **NASA Static Thermal Anomalies Mask + Detections (Mar 2025)** | official NASA product; global industrial / gas-flare / volcanic / landfill masking | **Directly substitutes D's industrial-masking moat.** Our `industrial_coverage.py` is now a reimplementation of a first-party layer released 13 months ago |
| **GEE `FIRMS` + `NASA/LANCE/SNPP_VIIRS/C2`** | Google-maintained, planetary-scale, continuous, free for research | The default for academic + climate-risk devs; 3-line import |
| **awesome-gee-community-catalog archival FIRMS vectors (Samapriya Roy)** | actively curated catalog | De-facto "clean archival substrate" for OSS geo devs |
| **pyronear/pyro-risks `NASAFIRMS` classes** | 27 stars; last push 2024-08-19 (20 mo stale) | Funded NGO couldn't sustain FIRMS wrapper maintenance |
| **datadesk/nasa-wildfires** | 34 stars; maintained 2025-06-26 | Thin wrapper; drift issues #15/24/39 **closed** by upstream |
| **Microsoft Power Automate NASA FIRMS connector** | Microsoft-authored | Enterprise low-code lane owned |

## Strongest attacks

### Attack 1 — NASA closed the industrial-masking moat in March 2025. **FATAL.**
[NASA Earthdata blog — FIRMS releases STA Mask + STA Detections](https://www.earthdata.nasa.gov/news/blog/firms-releases-new-features-identify-active-fires-type) (March 2025): global detection-tagging for "fires caused by burning vegetation" vs. "fires from natural heat sources or industrial heat sources" — gas flares, volcanic activity, cement plants, landfills. Phase 1's claim that "no OSS library provides global industrial masking" is now *true but irrelevant* — first-party does. D's single most-cited differentiator is obsolete before the library is written.

### Attack 2 — The "endpoint drift" narrative is a 2022–2023 artifact. **SERIOUS.**
FIRMS API is v4.0.66, stable, with `data_availability` and `missing_data` endpoints *built-in*. datadesk/nasa-wildfires issues #15/24/39 — the Phase 1 "fragile plumbing" exhibit — are all **closed**. Positioning a library around "surviving FIRMS churn" is fighting yesterday's war.

### Attack 3 — Audience is genuinely small, probably <200 active devs globally. **FATAL for "durable product."**
GitHub search across "nasa-firms" / "firms-api" / "firms-wrapper": ~10–15 serious repos total, half stale. Pyronear's `pyro-risks` at 27 stars hasn't been touched in 20 months — if a funded French NGO won't maintain their FIRMS wrapper, a solo non-profit dev certainly won't get paid attention either.

### Attack 4 — Competes with Vanyo's time on Candidate A. **DECISIVE.**
A needs AOI polling + LLM reasoning + notifications + persona story. D adds packaging, docs, issue triage, semver, deprecation windows, CI matrix across Python versions. Every hour on PyPI release notes is an hour not on A's agent loop — and A has a real user, D has "other developers."

### Attack 5 — GEE already *is* the substrate.
Any researcher with a GEE account gets `FIRMS` + `NASA/LANCE/SNPP_VIIRS/C2` + Samapriya's archival vectors as a 3-line import. Rebuilding outside GEE means rebuilding what Google maintains for free at petabyte scale. D is "the FIRMS lib for people without GEE" — a shrinking cohort.

## NASA roadmap signal

- **Mar 2025:** STA Mask + STA Detections released — closes industrial gap.
- **May–Nov 2025:** NASA ARSET Advanced FIRMS training (Parts 1 + 3). Active investment.
- **Jun 2025:** VIIRS Collection 2 375m Active Fire User Guide updated.
- **2025–2026:** Aerosol Index from NOAA-20/21 rolling into SMOKE/AEROSOLS. All Earth-science data sites migrating into Earthdata — consolidation, not fragmentation.

Every 6–12 months NASA eats another notch of the perceived OSS gap.

## Objections that didn't hold

- *"Drift-monitored ingest is still unique."* — `missing_data` + `data_availability` + stable v4 says no.
- *"Small audience still justifies publishing."* — true, but publish as **byproduct** of A, not as a thesis. That's 5 hours of docs, not a candidate.
- *"NASA could obsolete it."* — already partially happened.

## Net verdict

**Kill as standalone. Demote to "publish-as-byproduct-of-A."**

When A is built, extract whatever internal FIRMS + STA-consuming code is clean into a companion repo, publish to PyPI with honest positioning ("FIRMS + STA client extracted from Wildfire Nowcast"), document once, forget about it. No roadmap, no SLA, no donations story.

**Honest answer for anyone wanting a clean FIRMS + masking pipeline in April 2026:** NASA FIRMS API v4 + STA layers + GEE `FIRMS` for archive. That answer is sufficient. Candidate D was a solution to a problem NASA solved 13 months ago.
