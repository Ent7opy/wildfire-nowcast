"""Lightweight operational notification client.

Delivers alerts via webhook (Slack/Discord-compatible) and optional SMTP email.
All sends are fire-and-forget — never blocks the caller.
Rate-limited to at most one notification per event_type per window (default 900 s / 15 min).
Silently no-ops when no channel is configured.

Environment variables:
    NOTIFICATION_WEBHOOK_URL            Slack/Discord-compatible incoming webhook URL.
    NOTIFICATION_EMAIL_TO               Recipient address for email alerts.
    NOTIFICATION_SMTP_HOST              SMTP server hostname (required for email).
    NOTIFICATION_SMTP_PORT              SMTP port (default: 587).
    NOTIFICATION_SMTP_USER              SMTP username (optional).
    NOTIFICATION_SMTP_PASSWORD          SMTP password (optional).
    NOTIFICATION_EMAIL_FROM             Sender address (default: wildfire-nowcast@localhost).
    NOTIFICATION_RATE_LIMIT_SECONDS     Per-event rate-limit window in seconds (default: 900).
    NOTIFICATION_BURST_CAP              Max distinct event_types per AOI per burst window (default: 3).
    NOTIFICATION_BURST_WINDOW_SECONDS   Burst-tracking window in seconds (default: 60).
"""
from __future__ import annotations

import asyncio
import logging
import os
import smtplib
import threading
import time
from email.message import EmailMessage
from typing import Any, Literal

import httpx

LOGGER = logging.getLogger(__name__)

_rate_limit_lock = threading.Lock()
_last_sent: dict[str, float] = {}

_BURST_CAP: int = int(os.getenv("NOTIFICATION_BURST_CAP", "3"))
_BURST_WINDOW_SECONDS: int = int(os.getenv("NOTIFICATION_BURST_WINDOW_SECONDS", "60"))

# aoi_id → list of (sent_at_monotonic, event_type)
_burst_tracker: dict[str, list[tuple[float, str]]] = {}
_burst_tracker_lock = threading.Lock()

# Infrastructure / digest event_type prefixes that are never burst-capped.
_BURST_EXEMPT_PREFIXES: tuple[str, ...] = (
    "ingest_job_failed",
    "data_stale_critical",
    "denoiser_drift_hard",
    "burst_digest:",
)


def _rate_limit_seconds() -> int:
    return int(os.getenv("NOTIFICATION_RATE_LIMIT_SECONDS", "900"))


def _is_rate_limited(event_type: str) -> bool:
    """Return True if event_type was sent within the rate-limit window. Thread-safe."""
    now = time.monotonic()
    window = _rate_limit_seconds()
    with _rate_limit_lock:
        last = _last_sent.get(event_type, float("-inf"))
        if now - last < window:
            return True
        _last_sent[event_type] = now
        return False


def _build_webhook_payload(
    event_type: str,
    title: str,
    body: str,
    severity: Literal["info", "warning", "critical"],
    context: dict[str, Any],
) -> dict[str, Any]:
    color = {"info": "#36a64f", "warning": "#ff9800", "critical": "#e53935"}.get(severity, "#888888")
    fields = [{"title": k, "value": str(v), "short": True} for k, v in context.items()]
    return {
        "attachments": [
            {
                "color": color,
                "title": f"[{severity.upper()}] {title}",
                "text": body,
                "fields": fields,
                "footer": "wildfire-nowcast",
                "ts": int(time.time()),
            }
        ]
    }


async def _post_webhook(url: str, payload: dict[str, Any]) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()


def _send_email_blocking(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str | None,
    smtp_password: str | None,
    from_addr: str,
    to_addr: str,
    subject: str,
    body: str,
) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)
    with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as smtp:
        smtp.ehlo()
        if smtp.has_extn("STARTTLS"):
            smtp.starttls()
            smtp.ehlo()
        if smtp_user and smtp_password:
            smtp.login(smtp_user, smtp_password)
        smtp.send_message(msg)


async def _dispatch(
    event_type: str,
    title: str,
    body: str,
    severity: Literal["info", "warning", "critical"],
    context: dict[str, Any],
    webhook_url: str,
    email_to: str,
    smtp_host: str,
) -> None:
    if webhook_url:
        try:
            payload = _build_webhook_payload(event_type, title, body, severity, context)
            await _post_webhook(webhook_url, payload)
        except Exception:
            LOGGER.exception("Webhook notification failed (non-fatal): event_type=%s", event_type)

    if email_to and smtp_host:
        try:
            smtp_port = int(os.getenv("NOTIFICATION_SMTP_PORT", "587"))
            smtp_user = os.getenv("NOTIFICATION_SMTP_USER") or None
            smtp_password = os.getenv("NOTIFICATION_SMTP_PASSWORD") or None
            from_addr = os.getenv("NOTIFICATION_EMAIL_FROM", "wildfire-nowcast@localhost").strip()
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: _send_email_blocking(
                    smtp_host=smtp_host,
                    smtp_port=smtp_port,
                    smtp_user=smtp_user,
                    smtp_password=smtp_password,
                    from_addr=from_addr,
                    to_addr=email_to,
                    subject=f"[{severity.upper()}] {title}",
                    body=body,
                ),
            )
        except Exception:
            LOGGER.exception("Email notification failed (non-fatal): event_type=%s", event_type)


def _schedule_digest(aoi_id: str, count: int) -> None:
    """Fire-and-forget a single burst-digest notification for *aoi_id*."""
    notify(
        event_type=f"burst_digest:{aoi_id}",
        title=f"Multiple alerts for AOI {aoi_id}",
        body=f"{count} simultaneous events detected. Check dashboard for details.",
        severity="warning",
        aoi_id=aoi_id,
        suppressed_count=count,
    )


def _check_burst(aoi_id: str | None, event_type: str) -> bool:
    """Return True if this event should be suppressed due to the per-AOI burst cap.

    Non-AOI events (aoi_id is None or empty) and infrastructure event types are
    never burst-capped and always return False.
    """
    if not aoi_id:
        return False

    if any(event_type.startswith(prefix) for prefix in _BURST_EXEMPT_PREFIXES):
        return False

    now = time.monotonic()
    with _burst_tracker_lock:
        entries = _burst_tracker.get(aoi_id, [])
        # Prune stale entries outside the window.
        entries = [(t, et) for (t, et) in entries if now - t < _BURST_WINDOW_SECONDS]
        _burst_tracker[aoi_id] = entries

        distinct_types = {et for (_, et) in entries}
        count = len(distinct_types)

        if count < _BURST_CAP:
            # Under cap — allow and record.
            entries.append((now, event_type))
            _burst_tracker[aoi_id] = entries
            return False

        if count == _BURST_CAP:
            # Exactly at cap — this event tips us over; record it then digest.
            entries.append((now, event_type))
            _burst_tracker[aoi_id] = entries
            total = len({et for (_, et) in entries})
            # Schedule digest outside the lock to avoid re-entrant locking.
            digest_count = total
        else:
            # Already over cap — suppress silently; digest already sent.
            return True

    # We reach here only when count == _BURST_CAP (first overflow).
    _schedule_digest(aoi_id, digest_count)
    return True


def notify(
    event_type: str,
    title: str,
    body: str,
    severity: Literal["info", "warning", "critical"] = "info",
    **context: Any,
) -> None:
    """Fire-and-forget operational notification. Never blocks the caller.

    Args:
        event_type: Stable identifier used for rate limiting, e.g. ``"ingest_job_failed:firms"``.
        title:      Short human-readable summary (shown as attachment title).
        body:       Longer description (shown as attachment text).
        severity:   ``"info"`` | ``"warning"`` | ``"critical"`` — controls colour coding.
        **context:  Arbitrary key=value pairs rendered as fields in the notification.

    Silently no-ops when ``NOTIFICATION_WEBHOOK_URL`` is unset and no email is configured.
    Rate-limited: at most one notification per *event_type* per ``NOTIFICATION_RATE_LIMIT_SECONDS``
    (default 900 s / 15 min).
    """
    webhook_url = os.getenv("NOTIFICATION_WEBHOOK_URL", "").strip()
    email_to = os.getenv("NOTIFICATION_EMAIL_TO", "").strip()
    smtp_host = os.getenv("NOTIFICATION_SMTP_HOST", "").strip()

    if not webhook_url and not (email_to and smtp_host):
        return  # No channel configured — silent no-op

    aoi_id: str | None = context.get("aoi_id") or None
    if _check_burst(aoi_id, event_type):
        LOGGER.debug("burst cap suppressed event_type=%s for aoi_id=%s", event_type, aoi_id)
        return

    if _is_rate_limited(event_type):
        LOGGER.debug("Notification suppressed by rate limit: %s", event_type)
        return

    async def _run() -> None:
        await _dispatch(
            event_type, title, body, severity, dict(context),
            webhook_url=webhook_url, email_to=email_to, smtp_host=smtp_host,
        )

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run())
    except RuntimeError:
        # No running event loop (sync caller) — dispatch in a daemon thread.
        def _thread() -> None:
            asyncio.run(_run())

        threading.Thread(target=_thread, daemon=True).start()
