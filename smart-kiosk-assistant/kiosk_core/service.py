import threading
import tempfile
from pathlib import Path
from uuid import uuid4

import sounddevice as sd
from fastapi import UploadFile

from kiosk_core.audio_session import BrowserStreamSession, FileAudioSession, MicrophoneSession, TextQuerySession
from kiosk_core.models import (
    BrowserWakeWordStartRequest,
    BrowserWakeWordChunkResponse,
    BrowserWakeWordSessionResponse,
    FileSessionStartRequest,
    SessionStartRequest,
    SessionStopResponse,
    WakeWordSessionStartRequest,
)
from kiosk_core.wakeword import BrowserWakeWordSession, WakeWordListener


class SessionService:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, MicrophoneSession | FileAudioSession | BrowserStreamSession] = {}
        self._active_session_id: str | None = None
        self._wakeword_sessions: dict[str, BrowserWakeWordSession] = {}

    def start_session(self, request: SessionStartRequest) -> dict[str, object]:
        with self._lock:
            if self._active_session_id is not None:
                active = self._sessions[self._active_session_id]
                if active.snapshot()["status"] in {"running", "stopping"}:
                    raise ValueError("Another microphone session is already active")

            session = MicrophoneSession(request=request, on_complete=self._on_session_complete)
            self._sessions[session.session_id] = session
            self._active_session_id = session.session_id
            session.start()
            return session.snapshot()

    def start_session_after_wakeword(self, request: WakeWordSessionStartRequest) -> dict[str, object]:
        listener = WakeWordListener(
            wakeword_model=request.wakeword_model,
            threshold=request.wakeword_threshold,
            vad_threshold=request.wakeword_vad_threshold,
            patience_frames=request.wakeword_patience_frames,
            inference_framework=request.wakeword_inference_framework,
        )
        try:
            detection = listener.listen(
                device=request.device,
                sample_rate=request.sample_rate,
                timeout_seconds=request.wakeword_timeout_seconds,
            )
        except TimeoutError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Wake-word listener failed: {exc}") from exc
        snapshot = self.start_session(request)
        snapshot["wakeword"] = {
            "model": detection.model,
            "detected_label": detection.detected_label,
            "score": detection.score,
            "elapsed_seconds": detection.elapsed_seconds,
        }
        return snapshot

    def start_text_session(self, request: SessionStartRequest, text: str) -> dict[str, object]:
        """Answer a typed question. Runs alongside any audio session without
        the single-active-session gate, since text turns are transient and
        never touch the microphone."""
        session = TextQuerySession(request=request, query_text=text, on_complete=self._on_session_complete)
        with self._lock:
            self._sessions[session.session_id] = session
        session.start()
        return session.snapshot()

    def start_file_session(self, request: FileSessionStartRequest, upload: UploadFile) -> dict[str, object]:
        suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
        with self._lock:
            if self._active_session_id is not None:
                active = self._sessions[self._active_session_id]
                if active.snapshot()["status"] in {"running", "stopping"}:
                    raise ValueError("Another audio session is already active")

            with tempfile.NamedTemporaryFile(prefix="kiosk-upload-", suffix=suffix, delete=False) as temp_file:
                temp_file.write(upload.file.read())
                temp_path = temp_file.name

            session = FileAudioSession(request=request, audio_file_path=temp_path, on_complete=self._on_session_complete)
            self._sessions[session.session_id] = session
            self._active_session_id = session.session_id
            session.start()
            return session.snapshot()

    def start_browser_wakeword_session(self, request: BrowserWakeWordStartRequest) -> BrowserWakeWordSessionResponse:
        wakeword_session_id = str(uuid4())
        session = BrowserWakeWordSession(
            wakeword_model=request.wakeword_model,
            threshold=request.wakeword_threshold,
            vad_threshold=request.wakeword_vad_threshold,
            patience_frames=request.wakeword_patience_frames,
            inference_framework=request.wakeword_inference_framework,
            sample_rate=request.sample_rate,
        )
        with self._lock:
            self._wakeword_sessions[wakeword_session_id] = session
        return BrowserWakeWordSessionResponse(
            wakeword_session_id=wakeword_session_id,
            status="listening",
        )

    def push_browser_wakeword_audio(self, wakeword_session_id: str, wav_bytes: bytes) -> BrowserWakeWordChunkResponse:
        with self._lock:
            session = self._wakeword_sessions.get(wakeword_session_id)
        if session is None:
            raise KeyError(f"Unknown wake-word session: {wakeword_session_id}")

        try:
            detected, label, score = session.process_wav_chunk(wav_bytes)
        except ValueError:
            raise
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Wake-word listener failed while processing chunk: {exc}") from exc
        return BrowserWakeWordChunkResponse(
            wakeword_session_id=wakeword_session_id,
            detected=detected,
            score=score,
            detected_label=label,
        )

    def stop_browser_wakeword_session(self, wakeword_session_id: str) -> BrowserWakeWordSessionResponse:
        with self._lock:
            if wakeword_session_id not in self._wakeword_sessions:
                raise KeyError(f"Unknown wake-word session: {wakeword_session_id}")
            self._wakeword_sessions.pop(wakeword_session_id, None)
        return BrowserWakeWordSessionResponse(
            wakeword_session_id=wakeword_session_id,
            status="stopped",
        )

    def stop_session(self, session_id: str) -> SessionStopResponse:
        session = self._get_session_obj(session_id)
        session.stop()
        snapshot = session.snapshot()
        return SessionStopResponse(
            session_id=session_id,
            status=str(snapshot["status"]),
            stop_requested_at=session.stop_requested_at,
        )

    def get_session(self, session_id: str) -> dict[str, object]:
        return self._get_session_obj(session_id).snapshot()

    def get_response_audio_path(self, session_id: str, index: int) -> str:
        """Return the filesystem path of a synthesized response-audio segment.

        Raises KeyError if the session or the requested segment does not exist.
        """
        snapshot = self._get_session_obj(session_id).snapshot()
        for segment in snapshot.get("tts_audio_segments", []):
            if int(segment.get("index", -1)) == int(index):
                audio_file = str(segment.get("audio_file", ""))
                if audio_file and Path(audio_file).is_file():
                    return audio_file
                raise KeyError(f"Audio file missing for session {session_id} segment {index}")
        raise KeyError(f"No response-audio segment {index} for session {session_id}")

    def list_sessions(self) -> list[dict[str, object]]:
        with self._lock:
            return [session.snapshot() for session in self._sessions.values()]

    def list_input_devices(self) -> list[dict[str, str | int]]:
        devices = []
        for index, device in enumerate(sd.query_devices()):
            if int(device["max_input_channels"]) <= 0:
                continue
            devices.append(
                {
                    "id": index,
                    "name": str(device["name"]),
                    "default_samplerate": int(device["default_samplerate"]),
                }
            )
        return devices

    def host_mic_available(self) -> bool:
        """True when the machine running kiosk-core has at least one usable
        audio input device. Used by the UI to decide whether to capture audio
        directly on the host (preferred) or fall back to browser-mic streaming
        (e.g. when kiosk-core runs on a remote/headless machine)."""
        try:
            return len(self.list_input_devices()) > 0
        except Exception:
            return False

    def start_stream_session(self, request: SessionStartRequest) -> dict[str, object]:
        with self._lock:
            if self._active_session_id is not None:
                active = self._sessions[self._active_session_id]
                if active.snapshot()["status"] in {"running", "stopping"}:
                    # Auto-stop the previous session — the user intentionally started
                    # a new turn (new mic press). On a single-user kiosk this is always
                    # the right behaviour; we never block the new request.
                    try:
                        active.stop(reason="superseded_by_new_session")
                    except Exception:
                        pass  # already stopped or stopping — continue
                self._active_session_id = None

            session = BrowserStreamSession(request=request, on_complete=self._on_session_complete)
            self._sessions[session.session_id] = session
            self._active_session_id = session.session_id
            session.start()
            return session.snapshot()

    def push_audio_chunk(self, session_id: str, wav_bytes: bytes) -> None:
        session = self._get_session_obj(session_id)
        if not isinstance(session, BrowserStreamSession):
            raise ValueError("Session is not a browser stream session")
        if session.snapshot()["status"] not in {"running", "stopping"}:
            raise ValueError("Session is not active")
        session.push_audio(wav_bytes)

    def push_pcm(self, session_id: str, pcm_bytes: bytes) -> None:
        """Feed raw int16 PCM from the hands-free WebSocket uplink to a session."""
        session = self._get_session_obj(session_id)
        if not isinstance(session, BrowserStreamSession):
            raise ValueError("Session is not a browser stream session")
        if session.snapshot()["status"] not in {"running", "stopping"}:
            raise ValueError("Session is not active")
        session.push_pcm(pcm_bytes)

    def mark_playback_started(self, session_id: str) -> None:
        """Record the client's first-answer-audio playback report for a session."""
        session = self._get_session_obj(session_id)
        session.mark_speaker_playback()

    def signal_stream_end(self, session_id: str) -> None:
        session = self._get_session_obj(session_id)
        if not isinstance(session, BrowserStreamSession):
            raise ValueError("Session is not a browser stream session")
        session.signal_end()

    def _get_session_obj(self, session_id: str) -> MicrophoneSession | FileAudioSession | BrowserStreamSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session: {session_id}")
        return session

    def _on_session_complete(self, session_id: str) -> None:
        with self._lock:
            if self._active_session_id == session_id:
                self._active_session_id = None