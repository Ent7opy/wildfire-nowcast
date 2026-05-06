# Production Secrets Guide

How to manage secrets for Wildfire Nowcast deployments on Vercel.

The pre-pivot Python/Docker stack and its secrets (POSTGRES_PASSWORD, INTERNAL_API_KEY, GEMINI_API_KEY, CDSAPI_KEY, SPREAD_MODEL_CATALOG_SIGNING_KEY, etc.) have been removed during the A' pivot (see `pm/decisions/0005-problem-chosen-a-prime.md`). Only the secrets used by the current Next.js / Drizzle / Neon / Vercel-cron stack are documented below.

## Secret Inventory

| Variable | Required | Rotation frequency | Notes |
|----------|----------|--------------------|-------|
| `DATABASE_URL` | Yes | Quarterly (or on Neon credential rotation) | Pooled Neon connection string ending in `-pooler` |
| `FIRMS_MAP_KEY` | Yes | Yearly / on compromise | NASA FIRMS API key |
| `CRON_SECRET` | Yes | Quarterly | Bearer token shared between GitHub Actions cron and `/api/aoi/poll` |
| `AI_GATEWAY_API_KEY` | Yes (for briefs) | On compromise | Vercel AI Gateway key for `@ai-sdk/google` |
| `RESEND_API_KEY` | Yes (for notifications) | Quarterly | Email dispatch via Resend |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Yes (for auth) | On compromise | Browser-facing Clerk key (public by design) |
| `CLERK_SECRET_KEY` | Yes (for auth) | Quarterly | Server-only Clerk key |

A canonical example with comments lives in `.env.example`.

## Injecting Secrets on Vercel

All production and preview secrets are set via the Vercel dashboard. The current Next.js app reads them from `process.env` at request time (or at build time for `NEXT_PUBLIC_*`).

1. Open the project in the Vercel dashboard.
2. Navigate to **Settings → Environment Variables**.
3. Add each secret with the appropriate scope: **Production**, **Preview**, **Development**, or any combination.
4. Trigger a redeploy for changes to take effect (Vercel does not hot-reload env vars).

Local development uses `.env.local` (gitignored). Copy `.env.example` and fill in the values you need.

## Injecting Secrets in GitHub Actions

The FIRMS poll workflow (`.github/workflows/firms-poll.yml`) needs `CRON_SECRET` so it can authenticate against `/api/aoi/poll`.

1. In the GitHub repo, go to **Settings → Secrets and variables → Actions**.
2. Add `CRON_SECRET` as a repository secret.
3. The same value must be set on Vercel (Production scope) so the route handler accepts the bearer token.

## Secret Rotation Procedure

### `DATABASE_URL` (Neon)

1. Reset the password in the Neon console (or rotate the role).
2. Copy the new pooled connection string (host ending in `-pooler`).
3. Update `DATABASE_URL` in Vercel for Production and Preview scopes.
4. Redeploy. Verify by hitting an AOI route — a 503 means the value is still missing or wrong.

### `FIRMS_MAP_KEY`

1. Request a new key at https://firms.modaps.eosdis.nasa.gov/api/.
2. Update `FIRMS_MAP_KEY` in Vercel and as a GitHub Actions secret if the workflow ever calls FIRMS directly.
3. Redeploy. Trigger the FIRMS poll workflow manually to verify.
4. Revoke the old key with NASA.

### `CRON_SECRET`

1. Generate a new value: `openssl rand -hex 32`.
2. Update **both** the GitHub Actions repo secret **and** the Vercel Production env var **in the same window** — they must match for the cron to authenticate.
3. Redeploy on Vercel, then re-run the FIRMS poll workflow to verify.

### `AI_GATEWAY_API_KEY`

1. Rotate the key in the Vercel AI Gateway dashboard.
2. Update the env var on the project.
3. Redeploy. Verify by triggering a brief generation.

### `RESEND_API_KEY`

1. Create a new API key in the Resend dashboard.
2. Update on Vercel.
3. Redeploy and trigger a test notification.
4. Delete the old key in Resend.

### Clerk keys (`CLERK_SECRET_KEY`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`)

1. Rotate via the Clerk dashboard. The publishable key is safe to expose to the browser; the secret key is not.
2. Update both env vars on Vercel.
3. Redeploy. Active sessions may need to re-authenticate.

## What NOT To Do

- **Never commit `.env.local` or any other `.env*` file other than `.env.example` to git.** `.gitignore` covers this; do not bypass it.
- **Never log secrets.** Do not interpolate env vars that contain keys into log statements.
- **Never share secrets in Slack, email, or issue trackers.** Use the Vercel and GitHub UIs.
- **Never use the same secret across environments.** Production, Preview, and local development must each have independent credentials.
- **Never disable pre-commit hooks if/when added.** Fix the underlying issue.
