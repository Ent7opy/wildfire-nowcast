from fastapi import FastAPI

app = FastAPI(title="Wildfire Nowcast API", version="0.1.0")


@app.get("/health", tags=["internal"])
async def healthcheck():
    """Simple health endpoint used by local dev and readiness checks."""
    return {"status": "ok"}

