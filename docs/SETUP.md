## Project Setup Guide (minimal)

This page focuses on first-time setup only (platform prerequisites, `.env`, Docker/WSL notes, and common fixes). For commands and day-to-day workflows, use `README.md` + `make help`.

---

## 1. Prerequisites

| Tool | Windows | macOS | Linux |
| --- | --- | --- | --- |
| **Git** | Install from [git-scm.com](https://git-scm.com/download/win) | `brew install git` | `sudo apt-get install git` (or distro equivalent) |
| **Python 3.11** | Install from [python.org](https://www.python.org/downloads/windows/) and enable “Add to PATH” | `brew install python@3.11` | `sudo apt-get install python3.11 python3.11-venv` |
| **uv** (Python env manager) | `py -3 -m pip install --upgrade uv` | `python3 -m pip install --upgrade uv` | `python3 -m pip install --upgrade uv` |
| **Docker Desktop / Engine** | [Docker Desktop](https://www.docker.com/products/docker-desktop/) | [Docker Desktop](https://www.docker.com/products/docker-desktop/) | Docker Engine (`sudo apt-get install docker.io`) |
| **GNU make** | `choco install make` *(Chocolatey)*<br/>or use WSL | `brew install make` (installs as `gmake`, symlink if desired) | `sudo apt-get install build-essential` |
| **ecCodes** (for `cfgrib`) | *(weather ingest)* Use WSL + `sudo apt-get install libeccodes0` | `brew install eccodes` | `sudo apt-get install libeccodes0` |

> Tip: after installing `make` on Windows via Chocolatey, restart your terminal so `make` is on `PATH`.

---

## 2. Clone and configure env vars

```bash
git clone https://github.com/<your-org>/wildfire-nowcast.git
cd wildfire-nowcast
```

Create an `.env` in the repo root (override defaults as needed):

```env
FIRMS_MAP_KEY=your_firms_api_key
POSTGRES_USER=wildfire
POSTGRES_PASSWORD=wildfire
POSTGRES_DB=wildfire
```

---

## 3. Install dependencies (per project venvs)

Run once after cloning, or whenever dependencies change:

```bash
make install
```

This runs `uv sync --dev` in `api/`, `ui/`, `ml/`, and `ingest/`.

---

## 4. Docker / WSL notes

- Full stack via Docker Compose: `docker compose up --build` (API, UI, Postgres/PostGIS, Redis). Use `docker compose down` to stop.
- Windows: WSL is recommended for smoother tooling (make, ecCodes). Install prerequisites inside WSL if you hit path/compile issues.

---

## 5. Troubleshooting

| Issue | Fix |
| --- | --- |
| `make: command not found` | Install GNU make (see prerequisites) and restart the shell. |
| `uv: command not found` | Re-install `uv` (`python -m pip install --upgrade uv`) and ensure it’s on `PATH`. |
| Docker commands fail | Ensure Docker Desktop/Engine is running; check you can run `docker compose ps`. |
| FIRMS ingest missing key | Ensure `.env` contains `FIRMS_MAP_KEY` and restart your shell. |
| cfgrib/ecCodes errors | Install ecCodes (see prerequisites) and restart your shell. |
| `Access is denied` deleting `.venv\lib64` on Windows | Delete the old `.venv` (or run `make clean-venv`) and re-run `make install`; avoid sharing venvs between WSL and Windows. |
| Permission denied on clean | On Windows, run the terminal as Administrator if files are read-only. |

---

## 6. References

- [`docs/README.md`](./README.md) – Docs hub and navigation.
- [`README.md`](../README.md) – Single-source quickstart and make commands.
- [`Makefile`](../Makefile) – Run `make help` for all targets.
- [`docs/dev-python-env.md`](./dev-python-env.md) – Python/uv details.
- [`infra/README.md`](../infra/README.md) – Docker/infra specifics.
- [`docs/ingest/ingest_firms.md`](./ingest/ingest_firms.md) – NASA FIRMS fire detection ingest.
- [`docs/ingest/ingest_weather.md`](./ingest/ingest_weather.md) – Weather ingest details.
- [`docs/ingest/ingest_dem.md`](./ingest/ingest_dem.md) – Copernicus DEM preprocessing.

