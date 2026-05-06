# Raw GitHub signals — 2026-04-21

Agent 03. Inline-delivered. Format: `[quote / issue title] — [URL] — [repo] — [status]`.

## Repos (name — stars — pushed — focus)

- forefireAPI/forefire — 80 — 2025-12-12 — C++ wildland fire spread, Corsica origin
- Orion-AI-Lab/wildfire_forecasting — 79 — 2022-09-01 — DL wildfire danger forecast
- aiformankind/wildfire-smoke-dataset — 72 — 2021-01-31 — smoke image dataset
- bcgov/wps — 65 — 2026-04-20 — BC Wildfire Predictive Services
- pyronear/pyro-vision — 65 — 2024-04-12 — CV lib for edge
- lautenberger/elmfire — 59 — 2026-03-15 — Eulerian level-set spread
- XC-Li/Parallel_CellularAutomaton_Wildfire — 57 — 2026-02-20 — CA spread mpi4py
- ECMWFCode4Earth/wildfire-forecasting — 57 — 2026-03-28 — DL danger forecast
- SebastianGer/WildfireSpreadTS — 55 — 2026-04-15 — spread dataset/benchmark
- mitrefireline/simfire — 50 — 2026-04-12 — RL wildfire sim
- mitrefireline/simharness — 18 — 2026-04-12 — RL harness for simfire
- datadesk/nasa-wildfires — 34 — 2025-06-26 — Python FIRMS downloader
- pyronear/pyro-risks — 27 — 2024-08-19 — wildfire risk DS
- pyronear/pyro-api — 25 — 2026-04-20 — alert mgmt API
- jjdabr/BPINN-Wildfire — 24 — 2026-03-14 — Bayesian PINN wildfire
- WorldWindEarth/wildfire — 18 — 2026-01-29 — WMT geo-browser
- ecmwf/caliver — 18 — 2023-08-25 — fire-danger calib/verif (R)
- HakamShams/LOAN — 18 — 2025-11-06 — location-aware normalization
- project-araia/WildfireGPT — 17 — 2025-10-07 — RAG LLM wildfire Q&A
- pyronear/pyro-engine — 15 — 2026-04-20 — edge detection on Pi
- pyronear/pyro-platform — 13 — 2025-11-28 — monitoring dashboard
- vannizhang/wildfire-viz-app — 11 — 2025-01-29 — ArcGIS Firefly viz
- ncsu-geoforall-lab/r.fire.spread — 11 — 2025-10-19 — GRASS GIS spread
- bcgov/fbp-go — 9 — 2025-11-19 — mobile fire behavior calc
- j-tenny/pyrothermel — 8 — 2026-01-21 — Python bindings for Behave
- jcla490/landfire-python — 7 — 2025-11-21 — LANDFIRE API client
- bcgov/wps-fire-perimeter — 6 — 2024-11-29 — perimeter from imagery
- bcgov/nr-bcws-wfnews — 6 — 2026-04-21 — BC wildfire public site
- fire2a/fire-analytics-qgis-processing-toolbox-plugin — 6 — 2026-01-29 — QGIS plugin
- IBM/predict-wildfire-intensity — 35 — archived / "no longer maintained"

## Issues / quotes

### Geography / data coverage
- "Create map outside of US" — https://github.com/mitrefireline/simfire/issues/36 — OPEN — CONUS lock, LANDFIRE dependency
- "Working with pixel size larger than 30m" — https://github.com/mitrefireline/simfire/issues/15 — OPEN
- "Can't Run ignitions East of -106 Longitude" — https://github.com/lautenberger/elmfire/issues/87 — OPEN — LANDFIRE 2.4 SW-only coverage

### Onboarding / reproducibility
- "Drop down button for langage selection" — https://github.com/pyronear/pyro-platform/issues/186 — OPEN — includes dev-env war story
- "Is there support for newer GPUs" — https://github.com/Orion-AI-Lab/wildfire_forecasting/issues/1 — CLOSED (self-patch)
- "Is there a data preprocessing script and inference example" — https://github.com/Orion-AI-Lab/wildfire_forecasting/issues/2 — OPEN, no response

### FIRMS pipeline fragility
- "Failure to Acquire Run time Wildfire Data" — https://github.com/datadesk/nasa-wildfires/issues/39 — OPEN
- "Failure to Acquire Wildfire Data" — https://github.com/datadesk/nasa-wildfires/issues/24 — CLOSED — round-robin + auto-retry
- "SSL Connection Error" — https://github.com/datadesk/nasa-wildfires/issues/15 — CLOSED — `verify=False` workaround
- "Is this data updating?" — https://github.com/datadesk/nasa-wildfires/issues/7 — CLOSED — "it is not"
- "Update MODIS and VIIRS CSV urls" — https://github.com/datadesk/nasa-wildfires/issues/8 — endpoint moved

### Historical / manual weather + fuel overrides
- "Use historical weather" — https://github.com/WorldWindEarth/wildfire/issues/53
- "Manual weather inputs" — https://github.com/WorldWindEarth/wildfire/issues/54
- "Override fuel models" — https://github.com/WorldWindEarth/wildfire/issues/55
- "Enhance Wildfire Diamond direction" — https://github.com/WorldWindEarth/wildfire/issues/56
- "Add Haul Chart to Fire Lookout" — https://github.com/WorldWindEarth/wildfire/issues/57

### Export / notification / ops plumbing
- "Export alerts as CSV" — https://github.com/pyronear/pyro-api/issues/521
- "Link to platform in notification" — https://github.com/pyronear/pyro-api/issues/513
- "Set recorded_at when recording detection" — https://github.com/pyronear/pyro-api/issues/510 — timestamp lineage problem
- "Updates acknowledge button per user role" — https://github.com/pyronear/pyro-platform/issues/197
- "Audio alarm when alert is received" — https://github.com/pyronear/pyro-platform/issues/174
- "Filter sequences to today" — https://github.com/pyronear/pyro-platform/issues/185
- "Cascade cam deletion cleanup" — https://github.com/pyronear/pyro-api/issues/558

### Spread-physics correctness
- "Spotting not allowing fire to cross 30m fuel breaks" — https://github.com/lautenberger/elmfire/issues/111 — silent bug, docs mismatch
- "WSMFEFF_LOW_MULT value" — https://github.com/lautenberger/elmfire/issues/117 — docs/implementation mismatch
- "INITIAL_ATTACK final fire acreage depends on TSTOP" — https://github.com/lautenberger/elmfire/issues/109
- "Camera stuck on single position, identical images" — https://github.com/pyronear/pyro-engine/issues/347 — edge data-freshness failure

### Perimeter generation repeat-work
- ksharonin/feds-benchmarking — FEDS-PEC benchmarking
- tgestabrook/Dixie-Fire-perimeter-interpolation
- hrotovb001/HighResFirePerimeter
- bcgov/wps-fire-perimeter

### UI basics
- "Bbox & image do not fit on large screens" — https://github.com/pyronear/pyro-platform/issues/181 — firefighters actively report operational UI bugs
- "2 user can control the same cam" — https://github.com/pyronear/pyro-platform/issues/262 — multi-user conflict

### LLM / AI-agent angle
- project-araia/WildfireGPT — 17 stars, pure RAG Q&A over literature, no tool use, no real-time
- ndoteddy/ndo-ai-agent-fire-hotspot — 0 stars — multi-LLM (Gemini+LLaMA+Qwen) on FIRMS for SEA; pattern in the air
- All major OSS wildfire projects: zero LLM / agent surface

### No-repo platforms (searched, empty)
- Watch Duty, EFFIS / Copernicus fire, CAL FIRE / NIFC, NASA FIRMS first-party — none found
