# `app/api/` type & input-validation audit

Date: 2026-05-07
Author: scout
Scope: every route handler under `app/api/` (14 `route.ts` files,
plus the `_lib/handle.ts` shared helper). Goal: find untyped
inputs (`any` / `as unknown` / loose `as`) and routes whose
request shape isn't validated.

## TL;DR

There is **nothing safe to tighten in this subtree.** No
`any` casts, no `as unknown` casts, no `eslint-disable
no-explicit-any` directives. Every body-bearing route already
runs through a Zod schema via `parseJson` (or, for the cron
poll route, an inline `bodySchema.safeParse`). Querystring
routes use literal-set checks that are tighter than a Zod
wrapper would be at this LOC budget.

The only meaningful unvalidated input is the **Clerk webhook
payload**, where `evt = verify(...) as ClerkUserEvent` trusts a
Svix-verified vendor payload. Replacing the cast with a Zod
parse would change behaviour (reject events whose shape
drifts) and is therefore out of scope per the chore rules.
Catalogued as B1 below.

**Shipped this PR:** none. Inventory only.

## Inventory

| Route | Method | Input | Validation |
|---|---|---|---|
| `/api/aoi` | GET | — | n/a |
| `/api/aoi` | POST | JSON body | Zod `aoiCreateSchema` via `parseJson` |
| `/api/aoi/[id]` | GET | path | n/a |
| `/api/aoi/[id]` | PATCH | JSON body | Zod `aoiUpdateSchema` |
| `/api/aoi/[id]` | DELETE | path | n/a |
| `/api/aoi/[id]/rules` | PUT | JSON body | Zod `rulesUpsertSchema` |
| `/api/aoi/[id]/export` | GET | `?format=` | literal check (`geojson` \| `markdown`) |
| `/api/aoi/poll` | POST | JSON body + bearer | inline Zod `bodySchema` (strict) |
| `/api/brief/[id]/share` | POST | path | n/a (no body) |
| `/api/brief/[id]/share` | DELETE | path | n/a |
| `/api/export/aois.geojson` | GET | — | n/a |
| `/api/export/briefs.csv` | GET | `?since=` | manual `Date` parse |
| `/api/me` | GET | — | n/a |
| `/api/notify/feedback/[token]` | GET | `?v=` | literal check (`yes` \| `no`) |
| `/api/notify/pause/[token]` | GET | path | token verified in `_lib/handle` |
| `/api/notify/snooze/[token]` | GET | path | token verified in `_lib/handle` |
| `/api/notify/unsubscribe/[token]` | GET | path | token verified in `_lib/handle` |
| `/api/webhooks/clerk` | POST | raw body + Svix | signature verified; **payload shape cast** (B1) |

**`any` count: 0 → 0.** No regression target needed.

## Brainstorm-only / out-of-scope

**B1 — Clerk webhook payload shape is `as ClerkUserEvent`
(`route.ts:101`).** After `Webhook.verify()` returns, the
result is cast to `ClerkUserEvent` rather than parsed. The
downstream switch (`evt.type`) handles three known event
types and a default branch; missing fields surface as
`undefined` and the helpers (`pickPrimaryEmail`,
`pickDisplayName`) tolerate that. Replacing the cast with a
Zod parse would (a) reject events whose shape drifts even
slightly (Clerk has historically added optional fields), (b)
require the parser to be permissive enough that the only thing
gained over the current type is a runtime check the existing
helpers already make. Net: behaviour change for no defect-class
caught. Skipped.

**B2 — `?since=` parsing in `briefs.csv` accepts any
`Date`-parseable string.** `new Date("2026-foo")` returns
Invalid Date and is rejected, but `new Date("2026-05-07
12:34:56Z")` is accepted even though the doc comment says
`YYYY-MM-DD`. Tightening to a regex would change behaviour
(reject inputs we currently accept). Vanyo is the only caller
today; flagging for a future export-UX pass.

**B3 — `_lib/handle.ts` is the shared backbone for the four
notify token endpoints.** Its inputs are typed as
`ActionRouteContext` and the token parsing is centralised — so
the four handler files have no per-file validation surface,
which is the right factoring. Confirmed; nothing to change.

**B4 — Response shapes are not formally typed.** Every route
returns `NextResponse.json(...)` whose payload is inferred
from a literal object. There is no shared `ApiResponse<T>`
contract or OpenAPI spec. This is intentional (per
`CLAUDE.md` "no dev-time scaffolding for hypothetical
futures"); the dashboard consumes responses via `fetch +
.json()` and re-parses with its own types. Flagging only
because a future "publish a public API" milestone would want
this; not a defect today.

**B5 — `app/api/me/route.ts` uses
`decodeRows<{...}>(result)` with a hand-written row-shape
generic.** This is the documented pattern for raw-SQL queries
on the dual-backend (Neon + PGlite) repository layer; the
generic is a structural assertion, not a cast away from a
known type. Idiomatic; leaving alone.

## Why this is a no-LOC PR

The brief is explicit: *"If a real bug surfaces, flag — don't
fix"* and *"≤200 net-added LOC. No behavioural changes."*
Every candidate change above is either (a) behaviour-changing
(B1, B2) or (b) zero-defect cleanup (B3-B5). The honest
output is this note plus the inventory, so a future audit
doesn't re-derive the same null result.

## Files reviewed

- `app/api/aoi/route.ts`
- `app/api/aoi/[id]/route.ts`
- `app/api/aoi/[id]/rules/route.ts`
- `app/api/aoi/[id]/export/route.ts`
- `app/api/aoi/poll/route.ts`
- `app/api/brief/[id]/share/route.ts`
- `app/api/export/aois.geojson/route.ts`
- `app/api/export/briefs.csv/route.ts`
- `app/api/me/route.ts`
- `app/api/notify/_lib/handle.ts` (referenced)
- `app/api/notify/feedback/[token]/route.ts`
- `app/api/notify/pause/[token]/route.ts`
- `app/api/notify/snooze/[token]/route.ts`
- `app/api/notify/unsubscribe/[token]/route.ts`
- `app/api/webhooks/clerk/route.ts`
- `lib/api/handlers.ts` (`parseJson`, `withDb`)
- `lib/validators/aoi.ts` (`aoiCreateSchema`,
  `aoiUpdateSchema`, `rulesUpsertSchema`)
