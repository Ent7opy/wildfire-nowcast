# Goal

Execute the tasks described in `TASK_QUEUE.json`.

## Task file (verbatim)

```json
[
  {
    "id": "task-dynamic-ingest-pipeline",
    "title": "Implement Asynchronous Dynamic Ingestion & Forecast Pipeline",
    "rationale": "Currently, the system requires manual pre-ingestion of weather and terrain data (e.g., via 'make prepare'), which limits global usability. If a user clicks a fire in an unprovisioned region (e.g., Asia), the forecast fails. We need a 'Just-In-Time' (JIT) approach where clicking a fire triggers an automated background ingestion and processing loop, providing global coverage without requiring a massive upfront database for every continent.",
    "status": "pending",
    "priority": "high",
    "context": {
      "bottlenecks": [
        "Weather ingestion (NOAA GFS) takes 30-60s due to GRIB2 parsing.",
        "Terrain (Copernicus DEM) requires one-time regional download (~10s).",
        "Synchronous API calls will timeout; background workers are mandatory."
      ],
      "architecture_goals": [
        "Use Redis/RQ (already in project dependencies) for task orchestration.",
        "Refactor /forecast/generate to be non-blocking (returns task_id).",
        "Implement a 'Status' endpoint for the UI to poll (or WebSocket).",
        "Persist ingested data to ensure subsequent clicks in the same region are instant."
      ]
    },
    "implementation_steps": [
      "Initialize Redis/RQ worker environment in the api/ subproject.",
      "Refactor ingest.weather_ingest and ingest.dem_preprocess to be callable as Python functions with specific bbox/time parameters.",
      "Update api/routes/forecast.py to check for data existence before triggering simulation.",
      "Implement a background coordinator that sequences: Ingest Terrain -> Ingest Weather -> Run Spread Model.",
      "Add a 'task_status' table to track JIT job progress for the frontend."
    ]
  },
  {
    "id": "task-global-spatial-refactor",
    "title": "Generalize Spatial Grid & Feature Handling for Global Coverage",
    "rationale": "To support a worldwide system, we must remove dependencies on predefined named regions (like 'balkans'). The grid specification and feature engineering logic should be purely BBox-driven to work anywhere on Earth.",
    "status": "pending",
    "priority": "high",
    "context": {
      "bottlenecks": [
        "Many scripts default to a 'balkans' region name.",
        "Feature builders currently look for region-specific folders."
      ],
      "architecture_goals": [
        "Refactor GridSpec to be instantiable from an arbitrary BBox.",
        "Update feature engineering paths to use coordinate-based naming rather than region names."
      ]
    },
    "implementation_steps": [
      "Modify ml/spread_features.py to derive GridSpec from BBox if region_name is None.",
      "Update terrain and weather lookup logic to handle arbitrary coordinate ranges.",
      "Ensure all downstream ML services (denoiser, spread) accept BBox-only requests."
    ]
  },
  {
    "id": "task-jit-weather-optimization",
    "title": "Optimize JIT Weather Ingestion for Small AOIs",
    "rationale": "A worldwide system cannot wait 60 seconds for a full GFS GRIB download. We need fire-centric weather 'patches' that load in under 5 seconds.",
    "status": "pending",
    "priority": "medium",
    "context": {
      "bottlenecks": [
        "NOAA GFS downloads are monolithic.",
        "GRIB-to-NetCDF conversion is CPU-intensive for large areas."
      ],
      "architecture_goals": [
        "Implement partial GRIB downloads using HTTP Range requests.",
        "Use grib-index or similar to extract only the BBox of interest."
      ]
    },
    "implementation_steps": [
      "Research and implement GFS partial download capability in ingest/weather_ingest.py.",
      "Create a 'patch' mode for weather ingestion that skips global processing.",
      "Benchmark retrieval times for 10km x 10km areas."
    ]
  },
  {
    "id": "task-async-status-ui",
    "title": "Implement UI Progress Feedback for Async Forecasts",
    "rationale": "A user-friendly worldwide system must communicate progress during the JIT ingestion phase (e.g., 'Downloading weather...', 'Building features...').",
    "status": "pending",
    "priority": "high",
    "context": {
      "bottlenecks": [
        "Streamlit UI currently blocks or times out during long operations."
      ],
      "architecture_goals": [
        "Use a progress bar or status toast in Streamlit.",
        "Implement a polling mechanism to the API status endpoint."
      ]
    },
    "implementation_steps": [
      "Add a status polling component to ui/app.py.",
      "Update the 'Forecast' button to trigger the async task and enter a waiting state.",
      "Display real-time logs/status updates from the task worker on the UI."
    ]
  }
]
```
