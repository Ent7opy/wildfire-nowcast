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
    _burst_tracker,
    _check_burst,
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


class TestBurstCap(unittest.TestCase):
    """Per-AOI burst cap suppresses excess simultaneous alerts."""

    def setUp(self):
        _last_sent.clear()
        _burst_tracker.clear()
        # Use a cap of 3 and a large window so tests are deterministic.
        notif._BURST_CAP = 3
        notif._BURST_WINDOW_SECONDS = 60

    def tearDown(self):
        _burst_tracker.clear()
        _last_sent.clear()
        # Restore module defaults (env may not be set in tests).
        notif._BURST_CAP = int(os.getenv("NOTIFICATION_BURST_CAP", "3"))
        notif._BURST_WINDOW_SECONDS = int(os.getenv("NOTIFICATION_BURST_WINDOW_SECONDS", "60"))

    # ------------------------------------------------------------------
    # _check_burst unit tests (no I/O, no webhook)
    # ------------------------------------------------------------------

    def test_burst_cap_allows_first_n_alerts(self):
        """First _BURST_CAP distinct event_types for the same aoi_id are allowed."""
        aoi = "aoi-allows"
        self.assertFalse(_check_burst(aoi, "new_ignition"))
        self.assertFalse(_check_burst(aoi, "perimeter_breach"))
        self.assertFalse(_check_burst(aoi, "perimeter_growth"))
        # Exactly at cap after third — still not suppressed until the 4th.
        # (The 4th is what triggers suppression.)

    def test_burst_cap_suppresses_nth_plus_one(self):
        """The (cap+1)th distinct event_type within the window is suppressed."""
        aoi = "aoi-suppress"
        _check_burst(aoi, "new_ignition")
        _check_burst(aoi, "perimeter_breach")
        _check_burst(aoi, "perimeter_growth")
        # 4th event tips over the cap → suppressed
        result = _check_burst(aoi, "spread_trajectory")
        self.assertTrue(result)

    def test_burst_cap_independent_per_aoi(self):
        """Burst counts are tracked independently per aoi_id.

        3 events on aoi-1 and 3 events on aoi-2 are each allowed (counts don't
        bleed across AOIs).  A 4th event on aoi-1 is suppressed without affecting
        aoi-3, which has only 1 event and must still be allowed.
        """
        for et in ("new_ignition", "perimeter_breach", "perimeter_growth"):
            self.assertFalse(_check_burst("aoi-indep-1", et))
            self.assertFalse(_check_burst("aoi-indep-2", et))
        # aoi-indep-1 is at cap — 4th event is suppressed.
        self.assertTrue(_check_burst("aoi-indep-1", "spread_trajectory"))
        # aoi-indep-3 has had no events — must not be affected by aoi-indep-1's cap.
        self.assertFalse(_check_burst("aoi-indep-3", "new_ignition"))

    def test_burst_cap_resets_after_window(self):
        """After the window expires, a fresh batch of alerts is allowed."""
        aoi = "aoi-reset"
        # Fill the window.
        _check_burst(aoi, "new_ignition")
        _check_burst(aoi, "perimeter_breach")
        _check_burst(aoi, "perimeter_growth")
        # Backdate all entries far into the past so they expire.
        expired_time = time.monotonic() - 99999
        notif._burst_tracker[aoi] = [
            (expired_time, et) for (_, et) in notif._burst_tracker[aoi]
        ]
        # First three of a fresh batch must be allowed.
        self.assertFalse(_check_burst(aoi, "new_ignition"))
        self.assertFalse(_check_burst(aoi, "perimeter_breach"))
        self.assertFalse(_check_burst(aoi, "perimeter_growth"))

    def test_burst_cap_skipped_for_none_aoi_id(self):
        """Events with no aoi_id are never burst-capped."""
        for _ in range(10):
            self.assertFalse(_check_burst(None, "new_ignition"))
            self.assertFalse(_check_burst("", "new_ignition"))

    def test_burst_cap_skipped_for_infrastructure_events(self):
        """Infrastructure event types are never burst-capped even with an aoi_id."""
        aoi = "aoi-infra"
        for _ in range(10):
            self.assertFalse(_check_burst(aoi, "ingest_job_failed"))
            self.assertFalse(_check_burst(aoi, "data_stale_critical"))
            self.assertFalse(_check_burst(aoi, "denoiser_drift_hard"))
            self.assertFalse(_check_burst(aoi, "burst_digest:aoi-infra"))

    # ------------------------------------------------------------------
    # Integration test: notify() + webhook dispatch
    # ------------------------------------------------------------------

    def test_digest_sent_on_cap_exceeded(self):
        """When cap is exceeded, a digest notification is dispatched with the correct event_type."""
        aoi = "aoi-digest"
        dispatched: list[str] = []

        async def run():
            _last_sent.clear()
            _burst_tracker.clear()
            with patch.dict(os.environ, {"NOTIFICATION_WEBHOOK_URL": "http://fake/hook"}):
                with patch("api.notifications._post_webhook", new_callable=AsyncMock) as mock_post:
                    mock_post.side_effect = lambda url, payload: dispatched.append(
                        payload["attachments"][0]["title"]
                    )
                    # Send cap events — all should proceed.
                    for et in ("new_ignition", "perimeter_breach", "perimeter_growth"):
                        notify(et, "Title", "Body", aoi_id=aoi)
                    await asyncio.sleep(0.1)
                    # 4th event tips over the cap → suppressed; digest fires instead.
                    notify("spread_trajectory", "Title", "Body", aoi_id=aoi)
                    await asyncio.sleep(0.1)

            # 3 real alerts + 1 digest (burst_digest:aoi-digest).
            self.assertEqual(len(dispatched), 4)
            digest_titles = [t for t in dispatched if "Multiple alerts" in t]
            self.assertEqual(len(digest_titles), 1)

        asyncio.run(run())
