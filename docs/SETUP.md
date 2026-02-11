# Setup

## Environment Contract

A runnable environment needs:

- Python 3.11+
- A relational store with spatial support
- A cache/queue service for background tasks
- Local disk or object storage for raster artifacts

## Local Workflow

From repository root:

```bash
make help
make install
make db-up
make migrate
make dev-api
make dev-ui
```

If your environment differs, preserve the same high-level behavior:

- API service available
- UI service available
- Data store reachable
- Background processing enabled
