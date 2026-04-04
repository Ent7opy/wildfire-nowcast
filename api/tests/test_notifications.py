"""Tests for api/notifications.py — webhook format, rate limiting, graceful degradation."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
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
    _post_webhook,
    _resolve_webhook_url,
    _sign_payload,
    _validate_webhook_url,
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


class TestResolveWebhookUrl(unittest.TestCase):
    """Unit tests for _resolve_webhook_url severity-based channel routing."""

    _ROUTING_VARS = (
        "NOTIFICATION_WEBHOOK_URL",
        "NOTIFICATION_WEBHOOK_URL_CRITICAL",
        "NOTIFICATION_WEBHOOK_URL_WARNING",
        "NOTIFICATION_WEBHOOK_URL_INFO",
        "NOTIFICATION_WEBHOOK_URL_CRITICAL_ONLY",
    )

    def _clean_env(self):
        """Remove all routing env vars to start from a known-blank state."""
        for var in self._ROUTING_VARS:
            os.environ.pop(var, None)

    def setUp(self):
        self._clean_env()

    def tearDown(self):
        self._clean_env()

    def test_resolve_webhook_url_returns_none_when_nothing_configured(self):
        """With no env vars set, _resolve_webhook_url returns None for all severities."""
        for severity in ("critical", "warning", "info"):
            self.assertIsNone(_resolve_webhook_url(severity))

    def test_critical_uses_severity_specific_url(self):
        """When NOTIFICATION_WEBHOOK_URL_CRITICAL is set, critical events use it."""
        with patch.dict(
            os.environ,
            {
                "NOTIFICATION_WEBHOOK_URL_CRITICAL": "http://critical-channel/hook",
                "NOTIFICATION_WEBHOOK_URL": "http://fallback/hook",
            },
        ):
            self.assertEqual(_resolve_webhook_url("critical"), "http://critical-channel/hook")

    def test_severity_specific_takes_precedence_over_fallback(self):
        """Severity-specific URL wins over the global fallback when both are set."""
        with patch.dict(
            os.environ,
            {
                "NOTIFICATION_WEBHOOK_URL_CRITICAL": "http://critical-specific/hook",
                "NOTIFICATION_WEBHOOK_URL": "http://fallback/hook",
            },
        ):
            result = _resolve_webhook_url("critical")
            self.assertEqual(result, "http://critical-specific/hook")
            self.assertNotEqual(result, "http://fallback/hook")

    def test_warning_falls_back_to_global_when_no_specific(self):
        """When only the global fallback is set, warning events use it."""
        with patch.dict(os.environ, {"NOTIFICATION_WEBHOOK_URL": "http://fallback/hook"}):
            self.assertEqual(_resolve_webhook_url("warning"), "http://fallback/hook")

    def test_info_suppressed_in_critical_only_mode(self):
        """In critical-only mode, info events return None (webhook suppressed)."""
        with patch.dict(
            os.environ,
            {
                "NOTIFICATION_WEBHOOK_URL": "http://fallback/hook",
                "NOTIFICATION_WEBHOOK_URL_CRITICAL_ONLY": "true",
            },
        ):
            self.assertIsNone(_resolve_webhook_url("info"))

    def test_warning_suppressed_in_critical_only_mode(self):
        """In critical-only mode, warning events return None (webhook suppressed)."""
        with patch.dict(
            os.environ,
            {
                "NOTIFICATION_WEBHOOK_URL": "http://fallback/hook",
                "NOTIFICATION_WEBHOOK_URL_CRITICAL_ONLY": "true",
            },
        ):
            self.assertIsNone(_resolve_webhook_url("warning"))

    def test_critical_still_sends_in_critical_only_mode(self):
        """In critical-only mode, critical events still use the fallback URL."""
        with patch.dict(
            os.environ,
            {
                "NOTIFICATION_WEBHOOK_URL": "http://fallback/hook",
                "NOTIFICATION_WEBHOOK_URL_CRITICAL_ONLY": "true",
            },
        ):
            self.assertEqual(_resolve_webhook_url("critical"), "http://fallback/hook")

    def test_critical_only_false_does_not_suppress(self):
        """NOTIFICATION_WEBHOOK_URL_CRITICAL_ONLY=false leaves normal fallback behaviour intact."""
        with patch.dict(
            os.environ,
            {
                "NOTIFICATION_WEBHOOK_URL": "http://fallback/hook",
                "NOTIFICATION_WEBHOOK_URL_CRITICAL_ONLY": "false",
            },
        ):
            self.assertEqual(_resolve_webhook_url("warning"), "http://fallback/hook")
            self.assertEqual(_resolve_webhook_url("info"), "http://fallback/hook")


class TestSeverityRoutingIntegration(unittest.TestCase):
    """Integration tests: notify() + _post_webhook with severity-based routing."""

    _ROUTING_VARS = (
        "NOTIFICATION_WEBHOOK_URL",
        "NOTIFICATION_WEBHOOK_URL_CRITICAL",
        "NOTIFICATION_WEBHOOK_URL_WARNING",
        "NOTIFICATION_WEBHOOK_URL_INFO",
        "NOTIFICATION_WEBHOOK_URL_CRITICAL_ONLY",
        "NOTIFICATION_EMAIL_TO",
        "NOTIFICATION_SMTP_HOST",
    )

    def setUp(self):
        _last_sent.clear()
        for var in self._ROUTING_VARS:
            os.environ.pop(var, None)

    def tearDown(self):
        _last_sent.clear()
        for var in self._ROUTING_VARS:
            os.environ.pop(var, None)

    def test_critical_webhook_posted_to_severity_specific_url(self):
        """notify() POSTs to NOTIFICATION_WEBHOOK_URL_CRITICAL when it is set."""
        posted_urls: list[str] = []

        async def run():
            _last_sent.clear()
            with patch.dict(
                os.environ,
                {
                    "NOTIFICATION_WEBHOOK_URL_CRITICAL": "http://critical/hook",
                    "NOTIFICATION_WEBHOOK_URL": "http://fallback/hook",
                },
            ):
                with patch("api.notifications._post_webhook", new_callable=AsyncMock) as mock_post:
                    mock_post.side_effect = lambda url, payload: posted_urls.append(url)
                    notify("ev_crit_routing", "Title", "Body", severity="critical")
                    await asyncio.sleep(0.05)

            self.assertEqual(len(posted_urls), 1)
            self.assertEqual(posted_urls[0], "http://critical/hook")

        asyncio.run(run())

    def test_info_suppressed_in_critical_only_mode_no_webhook_call(self):
        """In critical-only mode, info-severity notify() never calls _post_webhook."""
        async def run():
            _last_sent.clear()
            with patch.dict(
                os.environ,
                {
                    "NOTIFICATION_WEBHOOK_URL": "http://fallback/hook",
                    "NOTIFICATION_WEBHOOK_URL_CRITICAL_ONLY": "true",
                    # No email configured either, so notify() exits before dispatch.
                },
            ):
                with patch("api.notifications._post_webhook", new_callable=AsyncMock) as mock_post:
                    notify("ev_info_suppressed", "Title", "Body", severity="info")
                    await asyncio.sleep(0.05)
                    self.assertEqual(mock_post.call_count, 0)

        asyncio.run(run())

    def test_email_always_sends_regardless_of_webhook_routing(self):
        """Even when the webhook is suppressed (critical-only mode), email is still dispatched."""
        email_calls: list[dict] = []

        async def fake_email(**kwargs: object) -> None:
            email_calls.append(dict(kwargs))

        async def run():
            _last_sent.clear()
            with patch.dict(
                os.environ,
                {
                    "NOTIFICATION_WEBHOOK_URL": "http://fallback/hook",
                    "NOTIFICATION_WEBHOOK_URL_CRITICAL_ONLY": "true",
                    "NOTIFICATION_EMAIL_TO": "ops@example.com",
                    "NOTIFICATION_SMTP_HOST": "smtp.example.com",
                },
            ):
                with patch("api.notifications._post_webhook", new_callable=AsyncMock) as mock_post:
                    with patch(
                        "api.notifications._send_email_blocking",
                        side_effect=lambda **kw: email_calls.append(kw),
                    ):
                        notify("ev_email_always", "Title", "Body", severity="warning")
                        await asyncio.sleep(0.1)

                # Webhook must NOT have been called (suppressed).
                self.assertEqual(mock_post.call_count, 0)

            # Email must still have been dispatched.
            self.assertEqual(len(email_calls), 1)

        asyncio.run(run())


class TestValidateWebhookUrl(unittest.TestCase):
    """_validate_webhook_url enforces https:// for non-localhost URLs."""

    def test_https_url_accepted(self):
        url = "https://hooks.slack.com/services/T00/B00/xxx"
        self.assertEqual(_validate_webhook_url(url), url)

    def test_http_localhost_accepted(self):
        self.assertEqual(_validate_webhook_url("http://localhost:9/hook"), "http://localhost:9/hook")
        self.assertEqual(_validate_webhook_url("http://127.0.0.1:9/hook"), "http://127.0.0.1:9/hook")
        self.assertEqual(_validate_webhook_url("http://[::1]:9/hook"), "http://[::1]:9/hook")

    def test_http_non_localhost_rejected(self):
        self.assertIsNone(_validate_webhook_url("http://evil.example.com/hook"))

    def test_ftp_scheme_rejected(self):
        self.assertIsNone(_validate_webhook_url("ftp://example.com/hook"))

    def test_empty_string_rejected(self):
        self.assertIsNone(_validate_webhook_url(""))

    def test_no_scheme_rejected(self):
        self.assertIsNone(_validate_webhook_url("hooks.slack.com/services/T00/B00/xxx"))


class TestHmacSigning(unittest.TestCase):
    """Webhook requests include HMAC-SHA256 signatures when WEBHOOK_SECRET is set."""

    def test_sign_payload_deterministic(self):
        payload = b'{"key":"value"}'
        ts = "1700000000"
        secret = "test-secret"
        sig1 = _sign_payload(payload, ts, secret)
        sig2 = _sign_payload(payload, ts, secret)
        self.assertEqual(sig1, sig2)

    def test_sign_payload_matches_manual_hmac(self):
        payload = b'{"key":"value"}'
        ts = "1700000000"
        secret = "my-secret"
        expected = hmac.new(
            secret.encode(),
            f"{ts}.".encode() + payload,
            hashlib.sha256,
        ).hexdigest()
        self.assertEqual(_sign_payload(payload, ts, secret), expected)

    def test_different_secret_different_signature(self):
        payload = b'{"key":"value"}'
        ts = "1700000000"
        sig_a = _sign_payload(payload, ts, "secret-a")
        sig_b = _sign_payload(payload, ts, "secret-b")
        self.assertNotEqual(sig_a, sig_b)


class TestPostWebhookSecurity(unittest.TestCase):
    """_post_webhook adds security headers and validates URLs."""

    def test_signature_and_timestamp_headers_present_when_secret_set(self):
        """When WEBHOOK_SECRET is set, requests include X-Signature and X-Timestamp."""
        captured_headers: list[dict] = []

        async def run():
            with patch.dict(os.environ, {"WEBHOOK_SECRET": "test-secret-123"}):
                with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                    mock_resp = AsyncMock()
                    mock_resp.status_code = 200
                    mock_resp.raise_for_status = lambda: None
                    mock_post.return_value = mock_resp
                    await _post_webhook("https://hooks.slack.com/test", {"text": "hello"})
                    _, kwargs = mock_post.call_args
                    captured_headers.append(kwargs.get("headers", {}))

            headers = captured_headers[0]
            self.assertIn("X-Timestamp", headers)
            self.assertIn("X-Signature", headers)
            self.assertTrue(headers["X-Signature"].startswith("hmac-sha256="))

        asyncio.run(run())

    def test_no_signature_header_when_secret_unset(self):
        """When WEBHOOK_SECRET is not set, X-Signature header is absent."""
        captured_headers: list[dict] = []

        async def run():
            # Ensure WEBHOOK_SECRET is not set
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("WEBHOOK_SECRET", None)
                with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                    mock_resp = AsyncMock()
                    mock_resp.status_code = 200
                    mock_resp.raise_for_status = lambda: None
                    mock_post.return_value = mock_resp
                    await _post_webhook("https://hooks.slack.com/test", {"text": "hello"})
                    _, kwargs = mock_post.call_args
                    captured_headers.append(kwargs.get("headers", {}))

            headers = captured_headers[0]
            self.assertIn("X-Timestamp", headers)
            self.assertNotIn("X-Signature", headers)

        asyncio.run(run())

    def test_http_non_localhost_url_skipped(self):
        """_post_webhook does not send when URL is http:// to a non-localhost host."""
        async def run():
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                await _post_webhook("http://evil.example.com/hook", {"text": "hello"})
                mock_post.assert_not_called()

        asyncio.run(run())

    def test_timestamp_header_is_recent(self):
        """X-Timestamp should be a recent unix timestamp (within 5 seconds of now)."""
        captured_headers: list[dict] = []

        async def run():
            os.environ.pop("WEBHOOK_SECRET", None)
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_resp = AsyncMock()
                mock_resp.status_code = 200
                mock_resp.raise_for_status = lambda: None
                mock_post.return_value = mock_resp
                await _post_webhook("https://hooks.slack.com/test", {"text": "hello"})
                _, kwargs = mock_post.call_args
                captured_headers.append(kwargs.get("headers", {}))

            ts = int(captured_headers[0]["X-Timestamp"])
            now = int(time.time())
            self.assertAlmostEqual(ts, now, delta=5)

        asyncio.run(run())

    def test_webhook_response_status_logged(self):
        """Webhook POST response status code is logged."""
        async def run():
            os.environ.pop("WEBHOOK_SECRET", None)
            with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
                mock_resp = AsyncMock()
                mock_resp.status_code = 200
                mock_resp.raise_for_status = lambda: None
                mock_post.return_value = mock_resp
                with patch.object(notif.LOGGER, "info") as mock_log:
                    await _post_webhook("https://hooks.slack.com/test", {"text": "hello"})
                    mock_log.assert_called_once()
                    log_msg = mock_log.call_args[0][0]
                    self.assertIn("status=", log_msg)

        asyncio.run(run())
