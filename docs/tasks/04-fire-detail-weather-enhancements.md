# Task: Weather panel — follow-on enhancements

Follow-on work after the core weather panel (tasks 02 and 03) is live. Complete these in priority order.

---

## P1 — Near-term forecast (+6h, +12h)

### Goal

Extend the fire detail panel to show where weather conditions are headed over the next 6–12 hours, not just what they were at acquisition time. For ongoing incidents, this is more operationally useful than current conditions.

### Context

The weather ingest already stores GFS forecast horizons — read the `weather_runs` schema and ingest pipeline to understand what's available. The API and UI changes should follow the same patterns established in tasks 02 and 03.

The RH fire-risk level computation must apply to each forecast step, not just current conditions.

### Done when

- Fire detail API returns forecast conditions for at least +6h and +12h when GFS forecast data is available
- Each forecast step includes the same fields as the current-conditions block, including RH fire-risk level
- UI renders the forecast as a compact timeline beneath current conditions
- If forecast data is unavailable, current conditions still render normally — the forecast is additive

---

## P1 — Wind direction compass rose

### Goal

Replace or supplement the wind direction arrow with a compass rose that makes direction scannable at a glance for non-meteorologist users.

---

## P2 — MeteoAlarm warning integration (Europe)

### Goal

Surface official dangerous-weather warnings from MeteoAlarm alongside the weather block for fire detections in Europe. A fire under a RED wind warning tells a materially different operational story than the same fire with no active warnings — the user shouldn't have to look this up separately.

### Context

Read `docs/feature_meteoalarm_weather_warnings.md` for the full scope and design intent. The detail panel layout from tasks 02–03 should accommodate a `warnings` block alongside `weather` — don't design it in a way that makes this awkward to add later.

Outside Europe, this gracefully degrades to nothing shown. The data contract should be source-agnostic to support extending to NOAA (North America) or BoM (Australia) later.

---

## P2 — Higher-resolution weather source

### Goal

Improve weather accuracy for fires near complex terrain or coastlines by supporting a higher-resolution model as an optional backend alongside GFS.

### Context

The resolution note displayed in the current UI is intentional — it exists to manage expectations while GFS is the only source. Before pursuing this, validate that GFS error is materially affecting analyst decisions. Read the weather ingest architecture to understand what adding a second backend would require.
