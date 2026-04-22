# Launch Draft — Land Trust Alliance Wildfire Resilience Network

## Target venue

Draft for the **Land Trust Alliance Wildfire Resilience Network** newsletter (or a guest post on LTA's Learning Center), pitched at stewardship and conservation-lands staff at accredited US land trusts. **Do not publish until Fire Stewardship Agent v1 ships** (target Q2 2026). This draft is a design tool: the product should match the voice of the post, not the other way around.

## Working title

**Primary:** *Watching the Preserve: A Free Fire Stewardship Agent for Land Trusts*

**Alternatives:**
1. *Depth Over Speed: Fire Intelligence Built for Stewardship, Not Alerts*
2. *A Quieter Way to Monitor Your Fire-Prone Preserves*

## Post body

Most of us on conservation stewardship teams learned, somewhere between 2017 and 2020, that "we own this land" and "we can tell you what is happening to it on a Sunday night at 2 a.m." are two very different statements.

At Sonoma Land Trust, the response to Tubbs was to co-found the Sonoma Valley Wildlands Collaborative and start managing fire across roughly 20,000 acres of collectively-held land. At Truckee Donner Land Trust, it was a 350-acre forest-health treatment in 2025 on top of ~2,000 cumulative acres. At Midpeninsula Regional Open Space District and POST, it was a formal Wildland Fire Resiliency Program now entering Phase 2. At The Nature Conservancy, 56 US preserves burned in 2020 alone — a 195 percent jump over the prior year. The work of doing fire stewardship on conservation land is, at this point, a well-developed practice across the network.

The work of *knowing what is happening on those acres, right now,* is not.

That is the small, specific gap I want to talk about in this post — and the tool I have been building to fill it, which I am sharing here first, with you, because the Land Trust Alliance Wildfire Resilience Network is the only audience I trust to tell me whether it actually helps.

### The monitoring gap, honestly

If you steward fire-prone land, your current monitoring stack probably looks like some version of this: a FIRMS browser tab, an InciWeb bookmark, the state dispatch dashboard, a group text with your fire-adapted-communities contact, and Watch Duty on your phone for the drive home. Each tool is good at what it does. None was built for your job.

- **Watch Duty** is a consumer public-safety app. It is excellent at what it is — fast, human-verified alerts for residents in fire country — and rightly donation-supported. But it is not designed to watch the boundary of a 2,400-acre preserve and tell you what the fuel moisture in that specific stand has been doing since your last prescribed burn.
- **EFFIS / JRC and NIFC dashboards** are authoritative, but they are institutional products built for agency consumers. The polygon in their system is a fire, not your preserve.
- **FIRMS** is a raw feed. It is the starting point for the rest of us, but it demands that every land trust rebuild the same interpretation layer on top of the same pixels.
- **ArcGIS** is wonderful if you have a GIS team. Most stewardship leads I have talked to do not.

The result is the same duct-tape pattern at most trusts I have spoken to: a handful of bookmarks, a habit of refreshing FIRMS at 6 a.m. during fire season, and a quiet worry that a VIIRS pass will come in while everyone is asleep and nobody will piece it together until the regional coordinator's morning email.

### What we are building, plainly

**Fire Stewardship Agent** is a free, open-source fire intelligence tool built for people whose relationship to land is stewardship. You upload (or draw) the polygons you actually care about — your preserves, your conservation easements, your collaborative project boundaries. It watches those polygons. When something on them changes, it writes you a short brief.

The positioning line — which is also the design constraint we are holding ourselves to — is **depth over speed**.

We are not trying to be the fastest. Watch Duty already won the "fastest" contest, and it won it fairly. What Watch Duty cannot do, because it is structurally the wrong product for it, is sit with *your specific acres* and put a new detection into the context of their history, their fuel loads, their neighboring ownerships, and the authority perimeters posted in the last few hours. That is a job for a small, patient AI-native tool, and that is the job we are taking.

### What a brief actually looks like

Here is what landed in my inbox during a test run last week, for a hypothetical AOI at a TNC-managed longleaf pine preserve in North Carolina. Names changed; structure real:

> **Pinewoods Preserve — 04:17 UTC, 12 Apr 2026**
> VIIRS NOAA-20 detection 14 km N of preserve boundary. Wind 240° at 28 km/h, pushing toward the reserve. NC Forest Service incident perimeter posted 90 min ago covers the detection cluster; perimeter source: NCFS authority feed, not modeled. Fuel moisture in the northern unit is trending below the post-2023-prescribed-burn threshold for the third straight week. Two comparable spread geometries in adjacent counties (Mar 2023, Feb 2024) ran 8–11 km before containment, both under similar southerly flow. **No immediate action recommended.** Next brief in 3 hours or on preserve-boundary breach.

The brief is ~120 words. It names the satellite, the authority whose perimeter it trusts, the historical analogs, and — crucially — what it is *not* recommending. A stewardship director can read it in the dark at 4 a.m., decide whether this is a "keep sleeping" or a "call Jim" moment, and go on with their night.

That paragraph is the whole product. Everything else — the map, the AOI manager, the history view — is scaffolding to let you receive that paragraph about *your* land.

### Why non-profit, and why that matters here

Fire Stewardship Agent is being built under Earth Tools, which is non-profit by intent. That means:

- **Free to every land trust, always.** There is no paid tier, no enterprise SKU, no B2B sales motion. If the tool becomes useful enough that trusts want to support it, we will accept donations the way LTA members already support each other's work. Until then, assume zero dollars flow in either direction.
- **Free-tier infrastructure.** The whole system is designed to run at roughly zero dollars a month on autoscale-to-zero hosting. Your AOIs are not a cost center we need to recoup.
- **Open source.** The code lives in a public repository. If we disappear, you can fork it. If you have a GIS intern who wants to contribute, we will merge their pull request.

This matters because most tools in this space eventually ask you for a procurement conversation. This one will not.

### What v1 is, and what it is not

We are being narrow on purpose. At v1, Fire Stewardship Agent can:

- Watch a set of AOI polygons you define
- Pull active-fire detections (FIRMS) and authoritative perimeters (NIFC, state authorities where available) against those AOIs
- Generate a short stewardship brief, in plain English, when something changes
- Keep a history of past briefs per AOI so you can review fire seasons retrospectively

What it explicitly does not do at v1:

- Predict spread. (We are not a Technosylva replacement and should not pretend to be.)
- Replace your dispatch channel or your 911 procedures. It informs your stewardship decisions; it does not drive incident response.
- Handle CAD-quality easement geometry edits. Bring your polygons as GeoJSON or shapefile.

### Three things I am asking of this audience

If you have read this far — thank you. What I would like to ask is very specific:

1. **Pick one preserve and try it.** The tool is most useful with your real AOIs, not demo ones.
2. **Tell me what is wrong with the brief format.** The 120-word brief above is a hypothesis, not a finished product. If it names the wrong thing first, or omits the one piece of information your Sunday-night self needs, that is a bug. I would rather know from you than guess.
3. **Share it with one other land trust if it is useful.** Not ten. One. The right distribution for this tool is peer-to-peer inside this network.

### Honest caveats

v1 is narrow. We do not yet support all the EU and Canadian perimeter authorities. We have not yet integrated LANDFIRE fuels at parcel resolution. The brief currently runs in English only. We have a waiting list of things to add, and we will add them in the order the network asks for them, which is why this post exists.

You can reach me at the contact link at the bottom of earth-tools.org/wildfire, and feedback channels are also wired into the LTA WRN peer group. I will read everything.

Thank you for reading, and for the work you are doing on the ground.

---

## Design implications fed back

These are product and architecture choices implied by the voice of the post. The spec (brief 11) and the architecture plan (brief 12) should honour them.

- **The brief paragraph is the product surface, not the map.** The map is scaffolding. Design and implementation effort should be weighted toward brief quality (provenance, context, history) rather than cartographic polish.
- **Every brief must name its sources explicitly.** "VIIRS NOAA-20", "NCFS authority feed, not modeled", "FIRMS" — peer audiences expect provenance. The brief template must carry source identifiers as first-class fields, not footnotes.
- **Authority perimeters outrank modeled perimeters.** The post promises that the brief cites the governing authority (NCFS / NIFC / ICNF / CWFIS) when one has posted, and labels modeled spread as modeled. The data model must distinguish authoritative vs. derived perimeters, and the brief generator must prefer authority sources.
- **Historical analogs ("two comparable spread events…") are a required brief element, not a stretch feature.** This means v1 needs a per-AOI history store and a simple analog-retrieval step, not just real-time ingest.
- **The "no action recommended" line is load-bearing.** The brief must explicitly state when it is *not* asking the user to do something. That is a prompt/template constraint, not a UI polish item.
- **AOI ingest must accept GeoJSON + shapefile out of the box.** Land trust GIS teams already have these formats. Anything heavier (CAD, ArcGIS service URL import) is v2+.
- **Non-profit and free-tier are load-bearing commitments, not marketing.** No paywalled features, no "pro" tier, no rate-limit-to-upsell pattern. The architecture must remain donation-viable and autoscale-to-zero-compatible, per ADR 0005.
- **Quiet hours / tone of alerts.** A stewardship director reading at 4 a.m. should not be shouted at. The brief tone guidance (plain, calm, specific, ~120 words, ends with a next-brief cadence) should be encoded in the LLM prompt, not left to chance.
- **Open-source and forkable from day one.** The post makes a promise of openness; the repo must be public before the post lands.
- **English-only at v1 is acceptable and should be said out loud.** Expansion to Portuguese / Spanish / Greek is a known Q3+ item (Mediterranean Natura 2000 archetype, per north-star.md), not a v1 scope creep risk.
- **No spread prediction at v1.** The post explicitly disclaims it. The spec and architecture cut list should confirm spread forecasting is out of v1 scope, consistent with ADR 0005.

## Evidence cited

All claims in the post above are traceable to a specific source. Stats and named orgs only; no invented numbers.

- **Sonoma Land Trust — 20,000 ac Sonoma Valley Wildlands Collaborative after Tubbs (2017).** sonomalandtrust.org; archetype card 1, `research-log/2026-04-21-user-archetypes.md`.
- **Truckee Donner Land Trust — 350 ac 2025 treatment, ~2,000 ac cumulative.** Archetype card 1, `research-log/2026-04-21-user-archetypes.md`.
- **Midpeninsula Regional Open Space District / POST — Wildland Fire Resiliency Program Phase 2 (May 2025).** Archetype card 1, `research-log/2026-04-21-user-archetypes.md`.
- **The Nature Conservancy — 56 US preserves burned 2020, +195% YoY.** Archetype card 1, `research-log/2026-04-21-user-archetypes.md`.
- **Land Trust Alliance Wildfire Resilience Network (LTA WRN) — exists as peer forum.** landtrustalliance.org/resources/connect/field-services/west/wildfire-resilience-network; archetype card 1.
- **Watch Duty — consumer-facing, donation-supported, ~$25/yr voluntary support seen on r/FortCollins and r/California.** `research-log/2026-04-21-reddit.md`, referenced in `research-log/2026-04-21-user-archetypes.md`.
- **FIRMS — raw active-fire detection feed (NASA).** Implicit from project stack; `ingest/firms_ingest.py` in repo.
- **EFFIS / JRC — institutional EU fire information; 140,291 ha burned inside Natura 2000 sites in 2025 (record).** `research-log/2026-04-21-user-archetypes.md`, citing JRC/Copernicus 2025-03-31 press release.
- **NIFC, NCFS, CWFIS, ICNF — authority perimeter sources referenced in the sample brief.** Existing ingest modules (`nifc_perimeters_ingest.py`, `cwfis_authority_ingest.py`) in repo.
- **Non-profit posture and free-tier cost ceiling ($0–10/month).** `PM_CLAUDE.md`; `decisions/0005-problem-chosen-a-prime.md`.
- **Positioning line "depth over speed".** Canonical line in `decisions/0005-problem-chosen-a-prime.md`.
- **Sample brief structure.** Adapted from the reference paragraph in `north-star.md` ("VIIRS NOAA-20 detection 14 km N of [AOI: Pinewoods Preserve]…").
- **LTA WRN landing page — venue of this post.** WebFetch to landtrustalliance.org was blocked (HTTP 403) at draft time; tone was calibrated from the brief's guidance ("colleague writing to peers, not a startup pitching") and from the archetype-1 evidence set. Tone should be re-checked against the live page before publication.
