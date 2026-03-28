# Agent Directives

## Zero-tolerance for mocking

Never use fake, dummy, or placeholder data unless explicitly requested. If the real data schema is missing, ask for it. If no authoritative source exists, stop and flag it.

## Hard stops are mandatory

Use `STOP` or `BLOCKER` when:
- An authoritative source is missing for a required input
- Feature contract mismatches between train and infer
- Geospatial alignment is invalid
- Data is fake or fabricated in a production path

Hard stops cannot be overridden by warnings or science debt items.

## Warn, don't bypass

`WARNING` is stage-aware and must include:
- mitigation action
- tracking ID
- target stage (usually `science_grade`)

A `WARNING` cannot replace a `STOP`/`BLOCKER`.

## Push back on shortcuts

If the user suggests a quick workaround that compromises scientific integrity, flag it:
> "We've had to rewrite this before because of shortcuts. Let's do it the real way now."

## Maturity stages

- `mvp_operational` — working end-to-end baseline
- `science_grade` — science-quality promotion target

See `SCIENCE_DEBT.md` for open stage-gap items.
See `docs/spread_gate_requirements.md` for the full gate specification.
