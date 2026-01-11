"""DB queries for export jobs."""
from __future__ import annotations
from typing import Any, Optional
from uuid import UUID
from sqlalchemy import text, bindparam
from sqlalchemy.dialects.postgresql import JSONB
from api.db import get_engine

def create_job(kind: str, request: dict[str, Any]) -> dict[str, Any]:
    stmt = text("""
        INSERT INTO export_jobs (kind, status, request)
        VALUES (:kind, 'queued', :request)
        RETURNING *
    """).bindparams(bindparam("request", type_=JSONB))
    
    with get_engine().begin() as conn:
        row = conn.execute(stmt, {"kind": kind, "request": request}).mappings().one()
    return dict(row)

def get_job(job_id: UUID) -> Optional[dict[str, Any]]:
    stmt = text("SELECT * FROM export_jobs WHERE id = :id")
    with get_engine().begin() as conn:
        row = conn.execute(stmt, {"id": job_id}).mappings().first()
    return dict(row) if row else None

def update_job_status(job_id: UUID, status: str, result: Optional[dict] = None, error: Optional[str] = None):
    # Set timestamps based on status transitions
    ts_update = ""
    if status == "running":
        ts_update = ", started_at = now()"
    elif status in ("succeeded", "failed"):
        ts_update = ", finished_at = now()"
        
    stmt = text(f"""
        UPDATE export_jobs
        SET status = :status, result = :result, error = :error, updated_at = now() {ts_update}
        WHERE id = :id
    """).bindparams(bindparam("result", type_=JSONB))
    
    with get_engine().begin() as conn:
        conn.execute(stmt, {"id": job_id, "status": status, "result": result, "error": error})
