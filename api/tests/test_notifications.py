"""Tests for api/notifications.py — webhook format, rate limiting, graceful degradation."""
from __future__ import annotations

import asyncio
import os
import time
import unittest
from unittest.mock import AsyncMock, patch

import api.notifications as notif
from api.notifications import (
    _build_webhook_payload,
    _dispatch,
    _is_rate_limited,
    _last_sent,
    notify,
)


class TestWebhookPayload(unittest.TestCase):
    """_build_webhook_payload produces a Slack-compatible attachment payload."""

    def test_critical_color_and_title(self):
        payload = _build_webhook_payload("ev", "My Title", "My body", "critical", {})
        att = payload["attachments"][0]
        self.assertEqual(att["color"], "#e53935")
        self.assertIn("[CRITICAL]", att["title"])
        self.assertIn("My Title", att["title"])

    def test_warning_color(self):
        att = _build_webhook_payload("ev", "t", "b", "warning", {})["attachments"][0]
        self.assertEqual(att["color"], "#ff9800")

    def test_info_color(self):
        att = _build_webhook_payload("ev", "t", "b", "info", {})["attachments"][0]
        self.assertEqual(att["color"], "#36a64f")

    def test_unknown_severity_fallback_color(self):
        att = _build_webhook_payload("ev", "t", "b", "debug", {})["attachments"][0]
        self.assertEqual(att["color"], "#888888")

    def test_body_in_text_field(self):
        att = _build_webhook_payload("ev", "t", "my body text", "info", {})["attachments"][0]
        self.assertEqual(att["text"], "my body text")

    def test_footer(self):
        att = _build_webhook_payload("ev", "t", "b", "info", {})["attachments"][0]
        self.assertEqual(att["footer"], "wildfire-nowcast")

    def test_context_becomes_fields(self):
        payload = _build_webhook_payload("ev", "t", "b", "critical", {"job": "firms", "code": 1})
        fields = {f["title"]: f["value"] for f in payload["attachments"][0]["fields"]}
        self.assertEqual(fields["job"], "firms")
        self.assertEqual(fields["code"], "1")  # values are str-coerced

    def test_empty_context_no_fields(self):
        att = _build_webhook_payload("ev", "t", "b", "info", {})["attachments"][0]
        self.assertEqual(att["fields"], [])

    def test_ts_is_integer(self):
        att = _build_webhook_payload("ev", "t", "b", "info", {})["attachments"][0]
        self.assertIsInstance(att["ts"], int)


class TestRateLimiting(unittest.TestCase):
    def setUp(self):
        _last_sent.clear()

    def test_first_call_allowed(self):
        self.assertFalse(_is_rate_limited("event_first"))

    def test_second_call_within_window_blocked(self):
        _is_rate_limited("event_double")
        self.assertTrue(_is_rate_limited("event_double"))

    def test_different_events_independent(self):
        _is_rate_limited("event_x")
        self.assertFalse(_is_rate_limited("event_y"))

    def test_rate_limit_respects_env_window(self):
        """With a 0-second window every call is allowed."""
        _last_sent.clear()
        with patch.dict(os.environ, {"NOTIFICATION_RATE_LIMIT_SECONDS": "0"}):
            _is_rate_limited("ev_zero")
            # Should NOT be rate-limited because window=0
            self.assertFalse(_is_rate_limited("ev_zero"))

    def test_expired_window_allows_resend(self):
        event = "event_expired"
        _is_rate_limited(event)  # mark as sent
        # Manually backdate well past any window
        notif._last_sent[event] = time.monotonic() - 99999
        self.assertFalse(_is_rate_limited(event))


class TestGracefulDegradation(unittest.TestCase):
    """notify() must never raise and must no-op when no channel is configured."""

    def setUp(self):
        _last_sent.clear()

    def _clear_notification_env(self):
        for key in ("NOTIFICATION_WEBHOOK_URL", "NOTIFICATION_EMAIL_TO", "NOTIFICATION_SMTP_HOST"):
            os.environ.pop(key, None)

    def test_no_config_is_silent(self):
        self._clear_notification_env()
        # Should not raise under any circumstances
        notify("ev_no_config", "Title", "Body", severity="critical", job="firms")

    def test_webhook_http_error_is_swallowed(self):
        """A failed POST must be caught and logged, not re-raised."""
        _last_sent.clear()
        with patch("api.notifications._post_webhook", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = Exception("connection refused")
            # _dispatch should not raise even when _post_webhook fails
            asyncio.run(
                _dispatch(
                    "ev_err", "Title", "Body", "critical", {},
                    webhook_url="http://localhost:9/hook",
                    email_to="",
                    smtp_host="",
                )
            )

    def test_notify_does_not_raise_when_webhook_configured_and_fails(self):
        _last_sent.clear()
        with patch.dict(os.environ, {"NOTIFICATION_WEBHOOK_URL": "http://localhost:9/hook"}):
            with patch("api.notifications._post_webhook", new_callable=AsyncMock) as mock_post:
                mock_post.side_effect = Exception("network error")
                # notify() itself must not raise in sync context
                notify("ev_safe", "Title", "Body", severity="warning")
                # Give daemon thread time to finish
                time.sleep(0.05)


class TestNotifyDispatch(unittest.TestCase):
    """notify() actually calls _post_webhook when a webhook URL is set."""

    def setUp(self):
        _last_sent.clear()

    def test_webhook_called_in_async_context(self):
        captured: list[tuple] = []

        async def run():
            _last_sent.clear()
            with patch.dict(os.environ, {"NOTIFICATION_WEBHOOK_URL": "http://fake/hook"}):
                with patch("api.notifications._post_webhook", new_callable=AsyncMock) as mock_post:
                    mock_post.side_effect = lambda url, payload: captured.append((url, payload))
                    notify("ev_async", "Title", "Body", severity="critical", job="firms")
                    # Yield control so the scheduled task can run
                    await asyncio.sleep(0.05)

            self.assertEqual(len(captured), 1)
            url, payload = captured[0]
            self.assertEqual(url, "http://fake/hook")
            self.assertIn("attachments", payload)
            fields = {f["title"]: f["value"] for f in payload["attachments"][0]["fields"]}
            self.assertEqual(fields["job"], "firms")

        asyncio.run(run())

    def test_rate_limited_event_skips_webhook(self):
        _last_sent.clear()
        call_count = 0

        async def run():
            nonlocal call_count
            _last_sent.clear()
            with patch.dict(os.environ, {"NOTIFICATION_WEBHOOK_URL": "http://fake/hook"}):
                with patch("api.notifications._post_webhook", new_callable=AsyncMock) as mock_post:
                    mock_post.side_effect = lambda url, payload: None

                    notify("ev_rl_test", "Title", "Body")
                    await asyncio.sleep(0.05)
                    notify("ev_rl_test", "Title", "Body")  # should be rate-limited
                    await asyncio.sleep(0.05)

                    call_count = mock_post.call_count

        asyncio.run(run())
        self.assertEqual(call_count, 1)

    def test_notify_sync_context_spawns_thread(self):
        """In a sync context (no event loop), notify() must not raise."""
        _last_sent.clear()
        with patch.dict(os.environ, {"NOTIFICATION_WEBHOOK_URL": "http://fake/hook"}):
            with patch("api.notifications._post_webhook", new_callable=AsyncMock):
                notify("ev_sync", "Title", "Body")
                time.sleep(0.05)  # give the daemon thread time to finish
