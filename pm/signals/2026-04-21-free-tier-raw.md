# Free-tier raw pricing snapshots — 2026-04-21

Access date: 2026-04-21. Pricing drifts — re-check before committing.

## Vercel Hobby
- https://vercel.com/docs/plans/hobby (2026-04-21)
- Function Invocations: 1,000,000/mo
- Function Duration: 100 GB-hrs/mo
- Active CPU: 4 CPU-hrs/mo
- Provisioned Memory: 360 GB-hrs/mo
- Edge Requests: up to 1,000,000/mo
- Workflow Events: 50,000/mo; Workflow Data: 1 GB
- Build minutes: 6,000/mo
- Max function duration: 10s (configurable to 60s on Hobby)
- 100 deploys/day
- **Fair-use / non-commercial prohibition** — grey for donation-only non-profit

## Neon Free
- https://neon.com/docs/introduction/plans (2026-04-21)
- Storage: 0.5 GB/project
- Compute: 100 CU-hours/project/mo (~400 hrs at 0.25 CU)
- Autoscaling up to 2 CU (8 GB RAM)
- Scale-to-zero after 5 min (not disableable on free)
- Branches: 10/project; Projects: 100
- Egress: 5 GB/mo
- Metrics retention: 1 day
- Compute suspends on limit until upgrade

## Cloudflare Workers / R2 / D1
- https://developers.cloudflare.com/workers/platform/pricing/ (2026-04-21)
- Workers: 100k req/day, 10ms CPU/invocation
- KV: 100k reads/day, 1k writes/day, 1 GB stored
- R2: 10 GB-month, 1M Class A, 10M Class B ops/mo, egress FREE
- D1: 5M rows read/day, 100k rows written/day, 5 GB stored
- Durable Objects: 100k req/day, 13,000 GB-s/day (SQLite backend only on free)

## GitHub Actions
- https://docs.github.com/en/billing/managing-billing-for-your-products/managing-billing-for-github-actions/about-billing-for-github-actions (2026-04-21)
- Public repos: **unlimited standard-runner minutes**
- Private (GitHub Free): 2,000 min/mo
- Cron minimum interval: 5 minutes, best-effort (lags under load)

## Vercel AI Gateway
- https://vercel.com/docs/ai-gateway/pricing (2026-04-21)
- Free tier: $5/mo included credit per team
- Paid: zero-markup pass-through
- BYOK supported

## Gemini 2.5 (via AI Gateway or direct)
- https://ai.google.dev/gemini-api/docs/pricing (2026-04-21)
- 2.5 Flash: $0.30 / $2.50 per 1M in/out
- 2.5 Flash-Lite: $0.10 / $0.40 per 1M
- 2.5 Pro: $1.00 / $10.00 per 1M
- Direct API free tier usable for dev, not production scale

## Claude Haiku (for comparison)
- https://platform.claude.com/docs/en/about-claude/pricing (2026-04-21)
- Haiku 3.5: $0.80 / $4.00 per 1M
- Haiku 4.5: $1.00 / $5.00 per 1M (batch: $0.50 / $2.50)

## NASA FIRMS
- https://firms.modaps.eosdis.nasa.gov/api/area/ (2026-04-21)
- MAP_KEY limit: 5,000 transactions / 10-minute interval
- Area API: 1–5 consecutive days per request
- Archive API: wider window
- Transactions scale with payload size

## Railway (current baseline for comparison)
- https://railway.com/pricing (2026-04-21)
- Hobby: $5/mo minimum
- Memory: $0.00000386/GB-sec (~$10.01/GB-month 24/7)
- CPU: $0.00000772/vCPU-sec (~$20.02/vCPU-month 24/7)
- Volume: $0.156/GB-month
- Egress: $0.05/GB
- **Current WFN stack estimate:** api + worker + ingest_scheduler + Postgres + Redis + titiler + pg_tileserv ≈ $25–50/mo small-scale
