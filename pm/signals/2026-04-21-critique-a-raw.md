# Raw signals — Critique of Candidate A (2026-04-21)

Agent 07.

## Watch Duty 2025 annual report + 2026 trajectory

- Annual report (load-bearing) — https://www.watchduty.org/blog/2025-annual-report
  - FY25 revenue $11.4M (2× YoY)
  - $7.6M cash 2026-01-01
  - FY26 budget $13.3M, target $8M ARR + $9M donations
  - ~74 employees across North America + Europe, Feb 2026
  - $1M Ring partnership funding all-50-states + national expansion
  - FY26 roadmap: flooding as next hazard, "professional users" as named tier
  - Overstory partnership: "their customers can consume their data directly in Watch Duty"
- Wikipedia — https://en.wikipedia.org/wiki/Watch_Duty — 8M+ users
- Recent press — https://www.northbaybiz.com/2026/04/07/watch-duty-upgrades-tools-for-north-bay-emergencies
- Crunchbase — https://www.crunchbase.com/organization/watch-duty

## Watch Duty × Overstory (utility vegetation AOI use case, already shipping)

- Cited inside WD annual report
- Adjacent: Live-EO Treeline — https://www.live-eo.com/product/treeline (same lane, funded, separate)

## John Mills stance

- LinkedIn — https://www.linkedin.com/in/johnclarkemills — "Still not for sale"
- TED talk — https://www.watchduty.org/blog/watch-duty-cofounder-ceo-john-mills-ted-talk — civic-infra framing is ideological, unlikely to bend

## NASA FIRMS constraints

- MAP_KEY limit — https://firms.modaps.eosdis.nasa.gov/api/map_key/ (5k tx / 10 min)
- API overview — https://firms.modaps.eosdis.nasa.gov/api/
- Earthdata tools — https://www.earthdata.nasa.gov/data/tools/firms (federal, no commercial SLA)

## Free-tier limits referenced in infra math

- Vercel Hobby — https://vercel.com/docs/plans/hobby (10s timeout, no commercial use)
- Vercel function limits — https://vercel.com/docs/functions/limitations
- Vercel pricing — https://vercel.com/pricing (Pro $20/user/mo floor)
- Neon free — https://neon.com/docs/introduction/plans (100 CU-hrs, 0.5 GB, autosuspend 5 min)
- Neon pricing — https://neon.com/pricing
- Neon free-plan guide — https://neon.com/blog/how-to-make-the-most-of-neons-free-plan

## LLM pricing

- Gemini 2.5 Flash — https://ai.google.dev/gemini-api/docs/pricing ($0.30 in / $2.50 out per 1M)
- Rate confirmation — https://pricepertoken.com/pricing-page/model/google-gemini-2.5-flash

## ChatGPT Agent (attack dropped as speculative)

- Intro — https://openai.com/index/introducing-chatgpt-agent
- Help — https://help.openai.com/en/articles/11752874-chatgpt-agent
- Codex Automations (closest live monitoring pattern) — https://openai.com/codex/

## Fogos.pt (donation-ceiling precedent)

- Cloudflare blog — https://blog.cloudflare.com/wildfire-fogos-pt-portugal-ddos-attack/ — volunteer, unfunded, Project Galileo recipient
- Play Store — https://play.google.com/store/apps/details?id=com.tomahock.fogos
- ReliefWeb citation — https://reliefweb.int/report/portugal/portugal-wildfire-jrc-effis-anepc-fogos-pt-media-echo-daily-flash-23-september-2025

## Utility vegetation-management crowded lane

- Review — https://kyro.ai/blog/10-best-utility-vegetation-management-software
- Hitachi Energy Service Suite X — https://www.hitachienergy.com/products-and-solutions/asset-and-work-management/work-management/service-suite-x/vegetation-management
- Live-EO Treeline — https://www.live-eo.com/product/treeline

## Self-critique notes

- Attack 1 (WD expansion) is the load-bearing claim. If WD's European staff footprint is research-only rather than product-facing, the critique weakens. Annual-report language reads as product-facing but Vanyo should challenge this first if he wants to save A.
- Attack 5 (free-tier math) was computed for a naive per-AOI-per-hour arch. Reconciled with agent 09's bucket-coalesced + 5%-LLM-gate arch in the research log. Both architects are self-consistent; they modeled different architectures.
