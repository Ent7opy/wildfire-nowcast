# Project Structure

**Wildfire Nowcast & Forecast** is an AI-first web application for monitoring active wildfires and predicting short-term spread (roughly 24–72 hours) using open satellite data, weather, terrain, and machine learning. The app enables users to explore active fires on a map, inspect fires in context, view probabilistic spread forecasts with uncertainty, explore risk maps, and get short natural-language summaries for selected areas.

For detailed project information, architecture, and guidelines, see [README.md](README.md).

## Repository Structure

This repository is organized into top-level directories that separate concerns:

- **`api/`** – FastAPI backend application providing REST endpoints for fires, forecasts, risk maps, historical data, and AOI summaries.

- **`ui/`** – Streamlit web application providing the map-based interface, layer controls, filters, inspection tools, and summary generation UI.

- **`ml/`** – Machine learning models, training scripts, and experiments. Includes hotspot denoiser, spread forecasting model, probability calibration, weather bias correction, and fire-risk index components.

- **`ingest/`** – Data ingestion pipelines for pulling and processing FIRMS fire detections, weather forecasts, terrain data (DEM), and optional context layers (land cover, population, infrastructure).

- **`infra/`** – Infrastructure configuration including Docker/Docker Compose files, deployment scripts, CI/CD configs, and operational tooling.

Each directory may contain its own README, requirements, and configuration files as components are developed.

