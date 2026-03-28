# Task: Async-All-the-Way — `asyncpg` + SQLAlchemy Async Engine

**Location:** `db.py`, all `async def` route handlers
**Impact:** Low now, high at scale — concurrency ceiling
**Maturity target:** `science_grade`

## Problem

`db.py` uses `create_engine` (synchronous SQLAlchemy). When this engine is used inside `async def` FastAPI route handlers, each DB call blocks the event loop thread for the duration of the query. Under low concurrency this is invisible. Under sustained load (multiple concurrent map tile requests, archive replays, live nowcast fetches), the event loop saturates and request latency spikes.

This is explicitly flagged as fine for current load — do not treat it as urgent. It is a `science_grade` migration target.

## Proposed Solution

Replace the sync engine with `create_async_engine` from `sqlalchemy.ext.asyncio`:

```python
# db.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine, async_sessionmaker

_async_engine: AsyncEngine | None = None

def get_async_engine() -> AsyncEngine:
    global _async_engine
    if _async_engine is None:
        _async_engine = create_async_engine(
            settings.async_database_url,  # postgresql+asyncpg://...
            pool_size=10,
            max_overflow=20,
            pool_recycle=1800,
        )
    return _async_engine

AsyncSessionLocal = async_sessionmaker(get_async_engine(), expire_on_commit=False)
```

FastAPI dependency:

```python
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
```

Route handlers become:

```python
@router.get("/fires")
async def list_fires(db: AsyncSession = Depends(get_db), ...):
    result = await db.execute(select(FireDetection).where(...))
    ...
```

## Migration Path

This is a larger refactor — do it incrementally:

1. Add `asyncpg` to deps and set `ASYNC_DATABASE_URL` in `.env.example`
2. Create `get_async_engine()` alongside the existing sync engine (both live in `db.py`)
3. Migrate routes one router at a time, starting with `routes/fires.py` (highest traffic)
4. Migrate `FireRepository` (Task 02) to use `AsyncSession` — async repo methods
5. Remove sync engine once all routes are migrated
6. Update `make migrate` — Alembic migrations remain sync (this is intentional and correct)

## Acceptance Criteria

- [ ] `asyncpg` added to `pyproject.toml` in `api/`
- [ ] `ASYNC_DATABASE_URL` documented in `.env.example` with `postgresql+asyncpg://` scheme
- [ ] `get_async_engine()` and `AsyncSessionLocal` defined in `db.py`
- [ ] At least one route (suggest `GET /fires`) migrated to `AsyncSession`
- [ ] No blocking DB calls (`session.execute` without `await`) remain in migrated routes
- [ ] Alembic migrations continue to use the sync engine — explicitly documented as intentional
- [ ] Load test (e.g. `locust` or `wrk`) shows measurable concurrency improvement at ≥ 50 concurrent users

## Notes

- Alembic does NOT need to migrate to async — keep it sync. This is a common point of confusion.
- `asyncpg` is Python 3.11-compatible (confirmed for this project's pinned version)
- Workers (RQ) run in their own threads and can continue using the sync engine — do not migrate them
- This task depends on Task 03 (pool singleton) being done first — the async engine should also be a singleton behind `get_async_engine()`
- Flag as BLOCKER if the sync engine is found blocking the event loop during an active incident
