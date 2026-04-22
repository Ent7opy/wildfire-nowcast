# Adversarial Critique — Candidate A (Non-Profit AOI Agent)

**Agent:** 07
**Stance:** Kill (brief required steel-manning the attack).
**Verdict:** KILL candidate A in its ADR-0003-reframed form.

## Thesis being attacked

*A fire-aware AOI agent that lets conservation NGOs, Indigenous fire stewards, protected-area managers, small municipalities, researchers, WUI homeowners, diaspora, and journalists subscribe polygons in natural language and receive reasoned AI briefs when a detection threatens their place, delivered globally on $0–10/month free-tier infra, donation-funded.*

## Strongest attacks

### Attack 1 — Watch Duty is not "US/CA only" anymore. It is the incumbent for this product, now international, now B2B, now funded.
**Evidence:** [Watch Duty 2025 annual report](https://www.watchduty.org/blog/2025-annual-report):
- $7.6M cash on hand as of 2026-01-01; $11.4M FY25 revenue (2× YoY); FY26 budget $13.3M; forecast $8M ARR + $9M donations.
- **~74 employees across North America + Europe** as of Feb 2026. Europe is in the staff footprint today.
- $1M Ring partnership funding "all 50 states" + national expansion; FY26 roadmap names **flooding** as next hazard and **professional users** as a named tier.
- **Partnered with Overstory (satellite vegetation encroachment for utilities) — "their customers can consume their data directly in Watch Duty."** This is literally the utility-vegetation-management AOI use case in L1.

**Severity:** FATAL. Phase 1's F1 said WD is un-competeable in its lane; ADR 0003 escaped by moving A *out* of the consumer-alert lane into custom-AOI / non-profit archetypes. That escape route is closed. WD already has a paid Pro tier, is already doing utility AOIs via Overstory, is already in Europe, has 8M users of reservoir trust, and $13M FY26 spend authority.

**What would need to be true for A to survive:** the custom-AOI slice WD serves must be narrow enough that international non-consumer archetypes (Indigenous fire stewards in Brazil, NGOs in Angola, Greek municipalities, diaspora) are structurally uninterested in WD. Unsubstantiated; likely false.

### Attack 2 — The donation-only ceiling is already occupied by WD and Fogos.pt.
ADR 0003 cites WD's $25/yr voluntary donation model as precedent, but WD captured that ceiling with 8M users. Fogos.pt's precedent is worse: zero funding, zero revenue, Cloudflare-hosted volunteers — and it's the dominant PT fire tracker. A third solo-dev entrant competing for the *same* voluntary donations in overlapping geographies will receive a microscopic share.

**Severity:** SERIOUS. Donation revenue likely won't fund even the LLM floor at target scale.

### Attack 3 — The "reasoned AI brief" fails the strip-the-AI-layer test.
L1's own output description ("why it matters for their asset, what changed in last N hours, confidence caveats, link to map state") is a template. A threshold-triggered templated SMS ("FIRMS detection 14km NNE of AOI 'Evia Parcel', wind 240° at 28 km/h bearing toward polygon, confidence 0.83, 3 detections in 6h, [map]") delivers 90% of the value at 0% LLM cost. The backend already computes bearing, distance, confidence, diff. The LLM adds prose polish on a payload already computed.

The "without LLM users drown in FPs" defence is backwards: the FP filter is the denoiser + industrial mask + thresholds — those are pre-AI.

**Severity:** SERIOUS. Per ADR 0003 the AI layer must be non-decorative; if a template beats it, the candidate fails its own AI-first test.

**What would save it:** brief content that is non-templatable — multi-event cross-source synthesis. That is L2's job.

### Attack 4 — FIRMS is a load-bearing federal dependency with a hard transaction cap.
[FIRMS MAP_KEY limit](https://firms.modaps.eosdis.nasa.gov/api/map_key/): **5,000 transactions per 10-minute window**. Naive per-AOI on-demand at 50 users × 100 AOIs × hourly = 120k transactions/day → blows cap unless aggressively bucket-coalesced through a single shared key. FIRMS is NASA LANCE, federally funded — "free + unlimited forever" is an unpriced assumption under current US science-budget instability.

**Severity:** MANAGEABLE at v1 with bucket coalescing (which agent 09's architecture already designs for). STRUCTURAL risk around federal dependency remains.

### Attack 5 — Free-tier math breaks at target load if polling is per-AOI.
Per-AOI hourly polling: 100 × 720 = 72k/mo DB queries → Neon's 100 CU-hrs cap exhausts in ~4 days. If LLM gate runs on *every* poll: 72k × 2k in = ~$43/mo. **But agent 09's architecture specifies bucket coalescing and a gate that fires ~5% of polls**, producing 30k invocations/mo and ~$0.14/mo LLM. Agent 07's attack applies to a *naive* implementation, not the proposed one.

**Severity:** MANAGEABLE **if agent 09's architecture is followed strictly**. FATAL if implemented naively. Worth flagging — the gating layer is load-bearing for cost, which means it's load-bearing for the product's viability.

## Free-tier reality check (agent 07's version)

Agent 07's math assumed per-AOI hourly polling with an LLM call per poll. That configuration is genuinely infeasible ($43/mo LLM, Neon cap blown in 4 days). Agent 09's math assumed bucket-coalesced polling with a 5% LLM gate. That configuration lands at ~$0/mo. **The two architects are both right — they modeled different architectures.**

**The reconciliation:** the free-tier story survives *only if* the gating and bucket-coalescing disciplines are treated as load-bearing product requirements, not optimizations. Remove either and A dies on infra cost.

## Objections I could not substantiate

- **"OpenAI/Anthropic/Google will ship a generic geofenced alert agent."** ChatGPT Agent exists; geofenced-alert packaging is not shipped publicly. Speculative; dropped.
- **"John Mills will sell."** Stance holds ("still not for sale, still growing"). Attack 1 doesn't need this to land.
- **"Non-profit solo dev can't be trusted."** Fogos.pt refutes this. Weaker than expected.

## Net verdict — KILL

Fatal attack: **Watch Duty April 2026 is not the 2024 WD Phase 1 synthesized against.** 74 staff NA+Europe, Pro tier, Overstory B2B utility-AOI partnership, $13M FY26 spend. They will reach every archetype ADR 0003 listed before a solo-dev v1 ships.

Phase 1's F1 ("WD un-competeable in its lane") was right; Phase 1 *underestimated how wide "its lane" is as of 2026-04-21.*

## Rescope options PM should consider

1. **Drop A entirely; pivot to D + E.** WD's Overstory integration *proves* the durable product shape is *the normalized fire data layer consumed by larger products*. Let WD be the UI; be the substrate underneath.
2. **Reframe A as single-tenant self-hosted open-source** — "personal AOI tool, BYO Neon + Vercel + API key, one polygon, one user." Kills the free-tier-at-scale problem and the donation-funding problem. Small audience, honest scope. Weekend project, not a Q2 deliverable.
3. **Geographic narrow to regions WD structurally won't reach in 2026** (Balkans + Turkey, Lusophone Africa, Ukraine post-conflict). Commit to the local-trust grind like Fogos.pt did. This is Candidate C with a new name, not A.

**Recommendation to PM:** demote A from "recommended #1" to "rescoped — option 1 (D+E only) or option 2 (self-hosted)." Do not commit to ADR-0003 reframe as-written.
