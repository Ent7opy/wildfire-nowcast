"""Lightweight operational notification client.

Delivers alerts via webhook (Slack/Discord-compatible) and optional SMTP email.
All sends are fire-and-forget — never blocks the caller.
Rate-limited to at most one notification per event_type per window (default 900 s / 15 min).
Silently no-ops when no channel is configured.

Environment variables:
    NOTIFICATION_WEBHOOK_URL            Slack/Discord-compatible incoming webhook URL (fallback).
    NOTIFICATION_WEBHOOK_URL_CRITICAL   Webhook URL for critical-severity events only.
    NOTIFICATION_WEBHOOK_URL_WARNING    Webhook URL for warning-severity events only.
    NOTIFICATION_WEBHOOK_URL_INFO       Webhook URL for info-severity events only.
    NOTIFICATION_WEBHOOK_URL_CRITICAL_ONLY
                                        If "true", the fallback NOTIFICATION_WEBHOOK_URL is only
                                        used for critical events; info/warning events are dropped
                                        when no severity-specific URL is configured.
    WEBHOOK_SECRET                      HMAC-SHA256 signing secret for outgoing webhooks.
                                        When set, each request includes X-Signature and
                                        X-Timestamp headers for payload authenticity and
                                        replay protection.
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
import hashlib
import hmac
import json
import logging
import os
import smtplib
import threading
import time
from email.message import EmailMessage
from typing import Any, Literal
from urllib.parse import urlparse

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


# ROUTING ARCHITECTURE (mvp_operational → science_grade):
# Current implementation supports severity-based channel routing via env vars.
# Target: per-AOI channel configuration stored in the aois table, allowing
# incident commanders to configure separate Slack channels per AOI or fire event.
# See audit gap: "No routing architecture — all notifications go to one global webhook."


_LOCALHOST_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _validate_webhook_url(url: str) -> str | None:
    """Validate webhook URL scheme and return the URL if valid, else None.

    Rules:
    - ``https://`` is always accepted.
    - ``http://`` is only accepted for localhost targets (development).
    - All other schemes (or malformed URLs) are rejected with a warning log.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        LOGGER.warning("Webhook URL rejected: failed to parse URL")
        return None

    scheme = (parsed.scheme or "").lower()
    hostname = (parsed.hostname or "").lower()

    if scheme == "https":
        return url

    if scheme == "http" and hostname in _LOCALHOST_HOSTS:
        return url

    if scheme == "http":
        LOGGER.warning(
            "Webhook URL rejected: http:// is only allowed for localhost targets, "
            "got host=%r. Use https:// for non-local webhooks.",
            hostname,
        )
        return None

    LOGGER.warning("Webhook URL rejected: unsupported scheme %r (must be https://)", scheme)
    return None


def _get_webhook_secret() -> str | None:
    """Return WEBHOOK_SECRET from env, or None if unset."""
    secret = os.getenv("WEBHOOK_SECRET", "").strip()
    return secret if secret else None


def _sign_payload(payload_bytes: bytes, timestamp: str, secret: str) -> str:
    """Compute HMAC-SHA256 signature over ``timestamp.payload_bytes``."""
    message = f"{timestamp}.".encode() + payload_bytes
    return hmac.new(secret.encode(), message, hashlib.sha256).hexdigest()


def _resolve_webhook_url(severity: str) -> str | None:
    """Return the webhook URL to use for a given severity level.

    Checks severity-specific URL first, falls back to NOTIFICATION_WEBHOOK_URL.
    Returns None if no webhook is configured for this severity.
    """
    specific = {
        "critical": os.getenv("NOTIFICATION_WEBHOOK_URL_CRITICAL", "").strip(),
        "warning": os.getenv("NOTIFICATION_WEBHOOK_URL_WARNING", "").strip(),
        "info": os.getenv("NOTIFICATION_WEBHOOK_URL_INFO", "").strip(),
    }.get(severity, "")

    if specific:
        return specific

    critical_only = os.getenv("NOTIFICATION_WEBHOOK_URL_CRITICAL_ONLY", "").strip().lower() == "true"
    if critical_only and severity != "critical":
        return None  # suppress non-critical when in critical-only mode

    fallback = os.getenv("NOTIFICATION_WEBHOOK_URL", "").strip()
    return fallback if fallback else None


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
    validated_url = _validate_webhook_url(url)
    if validated_url is None:
        LOGGER.warning("Webhook send skipped: URL failed validation url=%s", url)
        return

    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    timestamp = str(int(time.time()))

    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "X-Timestamp": timestamp,
    }

    secret = _get_webhook_secret()
    if secret:
        signature = _sign_payload(payload_bytes, timestamp, secret)
        headers["X-Signature"] = f"hmac-sha256={signature}"

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(validated_url, content=payload_bytes, headers=headers)
        LOGGER.info(
            "Webhook POST status=%d url=%s",
            resp.status_code,
            validated_url,
        )
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
    webhook_url: str | None,
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
) -> bool:
    """Fire-and-forget operational notification. Never blocks the caller.

    Returns:
        True if the notification was dispatched (scheduled for delivery).
        False if it was suppressed by burst cap, rate limit, or missing channel config.
        Callers that maintain transition-gate state (e.g. spread_trajectory_watch) MUST
        check this return value and only advance gate state when True is returned.

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
    webhook_url = _resolve_webhook_url(severity)
    email_to = os.getenv("NOTIFICATION_EMAIL_TO", "").strip()
    smtp_host = os.getenv("NOTIFICATION_SMTP_HOST", "").strip()

    if not webhook_url and not (email_to and smtp_host):
        return False  # No channel configured — silent no-op

    aoi_id: str | None = context.get("aoi_id") or None
    if _check_burst(aoi_id, event_type):
        LOGGER.debug("burst cap suppressed event_type=%s for aoi_id=%s", event_type, aoi_id)
        return False

    if _is_rate_limited(event_type):
        LOGGER.debug("Notification suppressed by rate limit: %s", event_type)
        return False

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

    return True
