# Raw signals — Critique of Candidate D (2026-04-21)

Agent 08.

## NASA STA release (the killer signal)

- https://www.earthdata.nasa.gov/news/blog/firms-releases-new-features-identify-active-fires-type (March 2025)
  - FIRMS released STA Mask + STA Detections layers
  - Distinguishes vegetation fires vs industrial/natural heat sources
  - Gas flares, cement plants, landfills, volcanic activity globally tagged
- https://firms.modaps.eosdis.nasa.gov/descriptions/Static_Thermal_Anomalies_Detections.html

## FIRMS API maturity

- https://firms.modaps.eosdis.nasa.gov/api/ — v4.0.66 current
- Endpoints: `area`, `countries`, `country`, `data_availability`, `kml_fire_footprints`, `map_key`, `missing_data`
- https://www.earthdata.nasa.gov/ — consolidation destination
- ARSET 2025 Part 1 — https://earthdata.nasa.gov/s3fs-public/2025-11/AdvFIRMS_Part1_BB_MFC_DD_JO.pdf
- ARSET 2025 Part 3 Q&A — https://earthdata.nasa.gov/s3fs-public/2025-06/arset-firms2025-part3-qa.pdf

## Existing OSS FIRMS alternatives

| Project | URL | Stars | Last push |
|---|---|---|---|
| datadesk/nasa-wildfires | https://github.com/datadesk/nasa-wildfires | 34 | 2025-06-26 |
| pyronear/pyro-risks | https://github.com/pyronear/pyro-risks | 27 | 2024-08-19 (stale) |
| GEE FIRMS dataset | https://developers.google.com/earth-engine/datasets/catalog/FIRMS | — | Google-maintained |
| GEE NASA/LANCE/SNPP_VIIRS/C2 | https://developers.google.com/earth-engine/datasets/catalog/NASA_LANCE_SNPP_VIIRS_C2 | — | Google-maintained |
| awesome-gee-community-catalog FIRMS vectors | https://gee-community-catalog.org/projects/firms_vector/ | — | Samapriya Roy |
| Microsoft NASA FIRMS connector | https://learn.microsoft.com/en-us/connectors/nasafirms/ | — | Microsoft |
| Call-for-Code/fires-api-nodejs | — | — | IBM-sponsored |
| mgrodecki/Wildfire-Digital-Twin | GitHub | — | 2026-04-12 |
| jakimovskidrago123-pixel/wildfire-gis-fastapi | GitHub | — | 2025-11-06 |
| Parthkk90/Torq | GitHub | — | 2026-03-07 |
| akshay070725/disaster-news-theme-blue | GitHub | — | 2026-02-12 |
| oscr104/FirmWare | GitHub | — | 2025-05-28 |

## Audience size evidence

- GitHub search "nasa-firms" / "firms-api" / "firms-wrapper": ~10–15 serious repos, half stale
- WildfireGPT 17 stars, datadesk 34, pyro-risks 27, Orion-AI 79 (slowing)
- Realistic audience: <200 active devs globally

## OSS maintainer burnout baseline

- Tidelift 2020: 46% of professional OSS maintainers burn out
- https://medium.com/@sohail_saifii/the-open-source-maintainer-burnout-crisis-nobodys-fixing-5cf4b459a72b
- https://roamingpigs.com/field-manual/open-source-maintainer-burnout/
- https://www.sonarsource.com/resources/library/open-source-maintainers/
- ~half of OSS projects solo-maintained, 60% unpaid

## Ecosystem notes

- VIIRS Collection 2 from SNPP + NOAA-20 + NOAA-21 is authoritative global feed
- EFFIS (Copernicus) accesses FIRMS via data request form — institutional consumers don't need a library
- FIRMS Aerosol Index adding to SMOKE/AEROSOLS — more first-party value
