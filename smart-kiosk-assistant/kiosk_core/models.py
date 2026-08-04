from datetime import datetime

from pydantic import BaseModel, Field

from kiosk_core import config


class SessionStartRequest(BaseModel):
    device: int | str | None = None
    sample_rate: int = Field(default=config.DEFAULT_SAMPLE_RATE, ge=8000, le=48000)
    chunk_seconds: float = Field(default=config.DEFAULT_CHUNK_SECONDS, gt=0.5, le=30)
    silence_timeout_seconds: float = Field(
        default=config.DEFAULT_SILENCE_TIMEOUT_SECONDS,
        gt=0.2,
        le=10,
    )
    max_session_seconds: float = Field(default=config.DEFAULT_MAX_SESSION_SECONDS, gt=1, le=300)
    silence_threshold: int = Field(default=config.DEFAULT_SILENCE_THRESHOLD, ge=1, le=32767)
    language: str | None = config.DEFAULT_ASR_LANGUAGE
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    analyzer_url: str = config.DEFAULT_ANALYZER_URL
    rag_url: str = config.DEFAULT_RAG_URL
    tts_url: str = config.DEFAULT_TTS_URL
    tts_model: str = config.DEFAULT_TTS_MODEL
    tts_voice: str | None = config.DEFAULT_TTS_VOICE
    tts_language: str | None = config.DEFAULT_TTS_LANGUAGE
    tts_instructions: str | None = config.DEFAULT_TTS_INSTRUCTIONS
    # Recent conversation turns prior to this question, oldest-first.
    # Forwarded verbatim to the RAG service so follow-ups have context.
    history: list[dict[str, str]] = Field(default_factory=list)
    # Persistent conversation session ID — reused across all voice turns in the
    # same customer conversation so the agent retains order state (order_id,
    # cart contents, etc.) between microphone presses.
    # The UI should generate this once per conversation and keep passing it.
    # If omitted, kiosk-core falls back to using the audio session UUID (old behaviour).
    conversation_id: str | None = None


class FileSessionStartRequest(SessionStartRequest):
    realtime_factor: float = Field(default=1.0, gt=0.0, le=100.0)


class WakeWordSessionStartRequest(SessionStartRequest):
    wakeword_model: str = Field(default=config.DEFAULT_WAKEWORD_MODEL)
    wakeword_threshold: float = Field(default=config.DEFAULT_WAKEWORD_THRESHOLD, ge=0.0, le=1.0)
    wakeword_vad_threshold: float = Field(default=config.DEFAULT_WAKEWORD_VAD_THRESHOLD, ge=0.0, le=1.0)
    wakeword_patience_frames: int = Field(default=config.DEFAULT_WAKEWORD_PATIENCE_FRAMES, ge=1, le=20)
    wakeword_timeout_seconds: float = Field(default=config.DEFAULT_WAKEWORD_TIMEOUT_SECONDS, ge=0.0, le=3600.0)
    wakeword_inference_framework: str = Field(default=config.DEFAULT_WAKEWORD_INFERENCE_FRAMEWORK)


class BrowserWakeWordStartRequest(BaseModel):
    sample_rate: int = Field(default=config.DEFAULT_SAMPLE_RATE, ge=8000, le=48000)
    wakeword_model: str = Field(default=config.DEFAULT_WAKEWORD_MODEL)
    wakeword_threshold: float = Field(default=config.DEFAULT_WAKEWORD_THRESHOLD, ge=0.0, le=1.0)
    wakeword_vad_threshold: float = Field(default=config.DEFAULT_WAKEWORD_VAD_THRESHOLD, ge=0.0, le=1.0)
    wakeword_patience_frames: int = Field(default=config.DEFAULT_WAKEWORD_PATIENCE_FRAMES, ge=1, le=20)
    wakeword_inference_framework: str = Field(default=config.DEFAULT_WAKEWORD_INFERENCE_FRAMEWORK)


class BrowserWakeWordSessionResponse(BaseModel):
    wakeword_session_id: str
    status: str


class BrowserWakeWordChunkResponse(BaseModel):
    wakeword_session_id: str
    detected: bool
    score: float
    detected_label: str | None = None


class SessionStopResponse(BaseModel):
    session_id: str
    status: str
    stop_requested_at: datetime
