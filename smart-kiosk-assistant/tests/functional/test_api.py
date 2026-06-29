"""
Tier 1 — API Functional Tests
==============================
Covers:
  NEX-T24260  Verify API health endpoint
  NEX-T24251  Verify microphone detection  (GET /api/v1/devices)
  NEX-T24266  Verify response interruption prevention  (409 on concurrent session)
  NEX-T24256  Verify conversation history retention  (model accepts history field)
  + defensive edge cases: session not found, empty audio body, EOS on unknown session,
    empty session list on fresh app, session listing after creation.

The `kiosk_app` fixture (defined in conftest.py) mounts a FastAPI TestClient
with `sounddevice` fully mocked.  No real audio device, Docker, or ML model
is required.

Run:
    pytest tests/functional/test_api.py -m tier1 -v
"""
import io
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_wav(duration_seconds: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Return a valid but silent 16-bit mono WAV blob."""
    import struct

    n_samples = int(sample_rate * duration_seconds)
    pcm = b"\x00\x00" * n_samples  # silent 16-bit samples

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# NEX-T24260 — Verify API health endpoint
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    """NEX-T24260 — curl http://127.0.0.1:8012/health → HTTP 200."""

    @pytest.mark.tier1
    def test_health_returns_200(self, kiosk_app):
        response = kiosk_app.get("/health")
        assert response.status_code == 200, (
            f"Expected 200 from /health, got {response.status_code}"
        )

    @pytest.mark.tier1
    def test_health_returns_ok_status(self, kiosk_app):
        response = kiosk_app.get("/health")
        body = response.json()
        assert body.get("status") == "ok", (
            f"Expected {{\"status\": \"ok\"}}, got {body}"
        )

    @pytest.mark.tier1
    def test_health_content_type_is_json(self, kiosk_app):
        response = kiosk_app.get("/health")
        assert "application/json" in response.headers.get("content-type", ""), (
            "Health endpoint must return application/json"
        )


# ---------------------------------------------------------------------------
# NEX-T24251 — Verify microphone detection
# ---------------------------------------------------------------------------
class TestDeviceEndpoint:
    """NEX-T24251 — Application should detect microphone without errors."""

    @pytest.mark.tier1
    def test_devices_returns_200(self, kiosk_app):
        response = kiosk_app.get("/api/v1/devices")
        assert response.status_code == 200, (
            f"Expected 200 from /api/v1/devices, got {response.status_code}"
        )

    @pytest.mark.tier1
    def test_devices_response_has_devices_key(self, kiosk_app):
        body = kiosk_app.get("/api/v1/devices").json()
        assert "devices" in body, f"Response missing 'devices' key: {body}"

    @pytest.mark.tier1
    def test_devices_returns_list(self, kiosk_app):
        body = kiosk_app.get("/api/v1/devices").json()
        assert isinstance(body["devices"], list), (
            f"'devices' must be a list, got {type(body['devices'])}"
        )

    @pytest.mark.tier1
    def test_devices_each_entry_has_required_fields(self, kiosk_app):
        """Each device entry must expose id, name, and default_samplerate."""
        devices = kiosk_app.get("/api/v1/devices").json()["devices"]
        for device in devices:
            assert "id" in device, f"Device entry missing 'id': {device}"
            assert "name" in device, f"Device entry missing 'name': {device}"
            assert "default_samplerate" in device, (
                f"Device entry missing 'default_samplerate': {device}"
            )


# ---------------------------------------------------------------------------
# Session listing — precondition for NEX-T24247 and general robustness
# ---------------------------------------------------------------------------
class TestSessionListing:
    """GET /api/v1/sessions — empty list on fresh app start."""

    @pytest.mark.tier1
    def test_list_sessions_returns_200(self, kiosk_app):
        response = kiosk_app.get("/api/v1/sessions")
        assert response.status_code == 200

    @pytest.mark.tier1
    def test_list_sessions_has_sessions_key(self, kiosk_app):
        body = kiosk_app.get("/api/v1/sessions").json()
        assert "sessions" in body, f"Response missing 'sessions' key: {body}"

    @pytest.mark.tier1
    def test_list_sessions_returns_list(self, kiosk_app):
        body = kiosk_app.get("/api/v1/sessions").json()
        assert isinstance(body["sessions"], list)


# ---------------------------------------------------------------------------
# Session not-found — defensive edge cases
# ---------------------------------------------------------------------------
class TestSessionNotFound:
    """Requests for unknown session IDs must return 404."""

    @pytest.mark.tier1
    def test_get_unknown_session_returns_404(self, kiosk_app):
        response = kiosk_app.get("/api/v1/sessions/non-existent-session-id")
        assert response.status_code == 404, (
            f"Expected 404 for unknown session, got {response.status_code}"
        )

    @pytest.mark.tier1
    def test_stop_unknown_session_returns_404(self, kiosk_app):
        response = kiosk_app.post("/api/v1/sessions/non-existent-session-id/stop")
        assert response.status_code == 404

    @pytest.mark.tier1
    def test_audio_end_on_unknown_session_returns_404(self, kiosk_app):
        """POST /api/v1/sessions/{id}/audio/end on unknown session → 404."""
        response = kiosk_app.post("/api/v1/sessions/non-existent-session-id/audio/end")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Empty audio body — defensive edge case
# ---------------------------------------------------------------------------
class TestEmptyAudioRejection:
    """POST /api/v1/sessions/{id}/audio with no body must return 400."""

    @pytest.mark.tier1
    def test_push_empty_audio_to_unknown_session_returns_404(self, kiosk_app):
        """Unknown session check takes priority over empty-body check."""
        response = kiosk_app.post(
            "/api/v1/sessions/non-existent-session-id/audio",
            content=b"",
            headers={"Content-Type": "application/octet-stream"},
        )
        # Service raises KeyError (unknown session) before checking body size
        assert response.status_code in {400, 404}

    @pytest.mark.tier1
    def test_push_empty_audio_body_rejected(self, kiosk_app):
        """Create a stream session then push an empty body — expect 400."""
        # Start a browser stream session (does not need sounddevice)
        with patch("kiosk_core.audio_session.AnalyzerClient"), \
             patch("kiosk_core.audio_session.RagClient"), \
             patch("kiosk_core.audio_session.TtsClient"):
            start_resp = kiosk_app.post(
                "/api/v1/sessions/start-stream",
                json={},
            )

        if start_resp.status_code != 200:
            pytest.skip(f"Could not start stream session: {start_resp.status_code}")

        session_id = start_resp.json()["session_id"]

        push_resp = kiosk_app.post(
            f"/api/v1/sessions/{session_id}/audio",
            content=b"",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert push_resp.status_code == 400, (
            f"Expected 400 for empty audio body, got {push_resp.status_code}: {push_resp.text}"
        )


# ---------------------------------------------------------------------------
# NEX-T24266 — Verify response interruption prevention
# ---------------------------------------------------------------------------
class TestSessionConflict:
    """
    NEX-T24266 — While assistant is answering, an additional ASK request
    must be rejected.  Maps to: starting a second session while one is
    already active → HTTP 409 Conflict.
    """

    @pytest.mark.tier1
    def test_second_stream_session_returns_409(self, kiosk_app):
        """
        Start a BrowserStreamSession, then immediately attempt to start
        another one.  The service must reject the second with 409.
        """
        with patch("kiosk_core.audio_session.AnalyzerClient"), \
             patch("kiosk_core.audio_session.RagClient"), \
             patch("kiosk_core.audio_session.TtsClient"):

            first = kiosk_app.post("/api/v1/sessions/start-stream", json={})
            assert first.status_code == 200, (
                f"First session start failed unexpectedly: {first.status_code} — {first.text}"
            )

            second = kiosk_app.post("/api/v1/sessions/start-stream", json={})

        assert second.status_code == 409, (
            f"Expected 409 Conflict when starting a second concurrent session, "
            f"got {second.status_code}: {second.text}"
        )

    @pytest.mark.tier1
    def test_conflict_response_contains_detail(self, kiosk_app):
        """409 response body must contain a human-readable 'detail' message."""
        with patch("kiosk_core.audio_session.AnalyzerClient"), \
             patch("kiosk_core.audio_session.RagClient"), \
             patch("kiosk_core.audio_session.TtsClient"):

            kiosk_app.post("/api/v1/sessions/start-stream", json={})
            second = kiosk_app.post("/api/v1/sessions/start-stream", json={})

        if second.status_code == 409:
            body = second.json()
            assert "detail" in body, f"409 response missing 'detail': {body}"


# ---------------------------------------------------------------------------
# NEX-T24256 — Conversation history retention (model-level validation)
# ---------------------------------------------------------------------------
class TestConversationHistory:
    """
    NEX-T24256 — Assistant should maintain context between interactions.
    Validates that SessionStartRequest correctly accepts a history payload
    so that multi-turn context can be forwarded to the RAG service.
    """

    @pytest.mark.tier1
    def test_session_start_accepts_history_field(self):
        """SessionStartRequest Pydantic model must accept a non-empty history list."""
        # Import the model directly — no sounddevice interaction needed
        from unittest.mock import MagicMock, patch

        sd_mock = MagicMock()
        sd_mock.query_devices.return_value = []

        with patch.dict("sys.modules", {"sounddevice": sd_mock}):
            import importlib
            import sys as _sys

            for mod in list(_sys.modules.keys()):
                if mod.startswith("kiosk_core"):
                    del _sys.modules[mod]

            from kiosk_core.models import SessionStartRequest

        history = [
            {"role": "user", "content": "What products are available?"},
            {"role": "assistant", "content": "We have Intel NUC, Arc GPU, and more."},
        ]
        request = SessionStartRequest(history=history)
        assert request.history == history, (
            f"History not preserved in model: {request.history}"
        )

    @pytest.mark.tier1
    def test_session_start_accepts_empty_history(self):
        """SessionStartRequest must accept an empty history list (default)."""
        from unittest.mock import MagicMock, patch

        sd_mock = MagicMock()
        sd_mock.query_devices.return_value = []

        with patch.dict("sys.modules", {"sounddevice": sd_mock}):
            import sys as _sys

            for mod in list(_sys.modules.keys()):
                if mod.startswith("kiosk_core"):
                    del _sys.modules[mod]

            from kiosk_core.models import SessionStartRequest

        request = SessionStartRequest()
        assert request.history == [], f"Default history must be [], got {request.history}"

    @pytest.mark.tier1
    def test_start_stream_endpoint_accepts_history_in_payload(self, kiosk_app):
        """POST /api/v1/sessions/start-stream with history payload must not return 422."""
        history = [
            {"role": "user", "content": "Tell me about the product"},
            {"role": "assistant", "content": "This is a smart kiosk"},
        ]
        with patch("kiosk_core.audio_session.AnalyzerClient"), \
             patch("kiosk_core.audio_session.RagClient"), \
             patch("kiosk_core.audio_session.TtsClient"):

            response = kiosk_app.post(
                "/api/v1/sessions/start-stream",
                json={"history": history},
            )

        assert response.status_code != 422, (
            f"Endpoint rejected valid history payload with 422: {response.text}"
        )
