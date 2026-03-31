"""Class-based repository for fire detection queries."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Iterable, Literal

from sqlalchemy.engine import Engine

import api.fires.repo as _repo

if TYPE_CHECKING:
    from sqlalchemy.engine import Connection


class FireRepository:
    """Thin class wrapper around the fire detection query module."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def validate_bbox(self, bbox: _repo.BBox) -> None:
        _repo.validate_bbox(bbox)

    def list_fire_detections_bbox_time(
        self,
        bbox: _repo.BBox,
        start_time: datetime,
        end_time: datetime,
        *,
        columns: Iterable[str] = ("lat", "lon", "acq_time"),
        limit: int | None = None,
        order: Literal["asc", "desc"] = "asc",
        include_noise: bool = False,
        include_masked: bool = False,
        min_confidence: float | None = None,
        min_fire_likelihood: float | None = None,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> dict:
        return _repo.list_fire_detections_bbox_time(
            bbox,
            start_time,
            end_time,
            columns=columns,
            limit=limit,
            order=order,
            include_noise=include_noise,
            include_masked=include_masked,
            min_confidence=min_confidence,
            min_fire_likelihood=min_fire_likelihood,
            cursor=cursor,
            offset=offset,
        )

    def list_fire_events_bbox_time(
        self,
        bbox: _repo.BBox,
        start_time: datetime,
        end_time: datetime,
        *,
        min_event_score: float | None = None,
        include_review_required: bool = True,
        limit: int = 1000,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> dict:
        return _repo.list_fire_events_bbox_time(
            bbox,
            start_time,
            end_time,
            min_event_score=min_event_score,
            include_review_required=include_review_required,
            limit=limit,
            cursor=cursor,
            offset=offset,
        )

    def list_fire_fronts_bbox_time(
        self,
        bbox: _repo.BBox,
        start_time: datetime,
        end_time: datetime,
        *,
        min_event_score: float | None = None,
        include_review_required: bool = True,
        limit: int = 2000,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> dict:
        return _repo.list_fire_fronts_bbox_time(
            bbox,
            start_time,
            end_time,
            min_event_score=min_event_score,
            include_review_required=include_review_required,
            limit=limit,
            cursor=cursor,
            offset=offset,
        )

    def get_fire_detection_by_id(self, detection_id: int) -> dict | None:
        return _repo.get_fire_detection_by_id(detection_id)

    # ------------------------------------------------------------------
    # Async methods — used by migrated async route handlers
    # ------------------------------------------------------------------

    async def async_list_fire_detections_bbox_time(
        self,
        bbox: _repo.BBox,
        start_time: datetime,
        end_time: datetime,
        *,
        columns: Iterable[str] = ("lat", "lon", "acq_time"),
        limit: int | None = None,
        order: Literal["asc", "desc"] = "asc",
        include_noise: bool = False,
        include_masked: bool = False,
        min_confidence: float | None = None,
        min_fire_likelihood: float | None = None,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> dict:
        return await _repo.async_list_fire_detections_bbox_time(
            bbox,
            start_time,
            end_time,
            columns=columns,
            limit=limit,
            order=order,
            include_noise=include_noise,
            include_masked=include_masked,
            min_confidence=min_confidence,
            min_fire_likelihood=min_fire_likelihood,
            cursor=cursor,
            offset=offset,
        )

    async def async_get_fire_detection_by_id(self, detection_id: int) -> dict | None:
        return await _repo.async_get_fire_detection_by_id(detection_id)

    async def async_list_fire_events_bbox_time(
        self,
        bbox: _repo.BBox,
        start_time: datetime,
        end_time: datetime,
        *,
        min_event_score: float | None = None,
        include_review_required: bool = True,
        limit: int = 1000,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> dict:
        return await _repo.async_list_fire_events_bbox_time(
            bbox,
            start_time,
            end_time,
            min_event_score=min_event_score,
            include_review_required=include_review_required,
            limit=limit,
            cursor=cursor,
            offset=offset,
        )

    async def async_list_fire_fronts_bbox_time(
        self,
        bbox: _repo.BBox,
        start_time: datetime,
        end_time: datetime,
        *,
        min_event_score: float | None = None,
        include_review_required: bool = True,
        limit: int = 2000,
        cursor: str | None = None,
        offset: int | None = None,
    ) -> dict:
        return await _repo.async_list_fire_fronts_bbox_time(
            bbox,
            start_time,
            end_time,
            min_event_score=min_event_score,
            include_review_required=include_review_required,
            limit=limit,
            cursor=cursor,
            offset=offset,
        )

    def get_fire_front_by_id(
        self,
        front_id: str,
        *,
        buffer_km: float = 0.0,
    ) -> dict | None:
        return _repo.get_fire_front_by_id(front_id, buffer_km=buffer_km)

    def update_false_source_masking(self, batch_id: int, conn: Connection | None = None) -> int:
        return _repo.update_false_source_masking(batch_id, conn)

    def update_persistence_scores(self, batch_id: int, conn: Connection | None = None) -> int:
        return _repo.update_persistence_scores(batch_id, conn)

    def update_landcover_scores(self, batch_id: int, conn: Connection | None = None) -> int:
        return _repo.update_landcover_scores(batch_id, conn)

    def update_weather_scores(self, batch_id: int, conn: Connection | None = None) -> int:
        return _repo.update_weather_scores(batch_id, conn)

    def update_fire_likelihood(self, batch_id: int, conn: Connection | None = None) -> int:
        return _repo.update_fire_likelihood(batch_id, conn)

    def update_all_scoring_for_batch(
        self,
        batch_id: int,
        conn: Connection | None = None,
    ) -> dict[str, int]:
        return _repo.update_all_scoring_for_batch(batch_id, conn)

    def get_latest_denoiser_gate_report(self) -> dict | None:
        return _repo.get_latest_denoiser_gate_report()

    def get_latest_denoiser_coverage_status(
        self, authority_profile: str = "wfigs_us"
    ) -> dict | None:
        return _repo.get_latest_denoiser_coverage_status(authority_profile)

    def get_latest_denoiser_industrial_coverage_status(
        self,
        source_profile: str | None = None,
        policy_version: str | None = None,
    ) -> dict | None:
        return _repo.get_latest_denoiser_industrial_coverage_status(
            source_profile, policy_version
        )

    def list_recent_denoiser_drift(self, limit: int = 50) -> list[dict]:
        return _repo.list_recent_denoiser_drift(limit)

    def list_denoiser_review_queue(self, limit: int = 200, status: str = "open") -> list[dict]:
        return _repo.list_denoiser_review_queue(limit, status)

    def resolve_denoiser_review_event(
        self,
        event_id: str,
        *,
        resolved_by: str,
        resolved_notes: str | None = None,
    ) -> int:
        return _repo.resolve_denoiser_review_event(
            event_id,
            resolved_by=resolved_by,
            resolved_notes=resolved_notes,
        )
