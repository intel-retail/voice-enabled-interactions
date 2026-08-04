import os


DEFAULT_ANALYZER_URL = os.getenv(
    "KIOSK_CORE_ANALYZER_URL",
    "http://127.0.0.1:8010/v1/audio/transcriptions",
)
DEFAULT_RAG_URL = os.getenv(
    "KIOSK_CORE_RAG_URL",
    "http://127.0.0.1:8020/api/v1/query",
)
DEFAULT_TTS_URL = os.getenv(
    "KIOSK_CORE_TTS_URL",
    "http://127.0.0.1:8011/v1/audio/speech",
)
DEFAULT_TTS_MODEL = os.getenv("KIOSK_CORE_TTS_MODEL", "qwen-tts")
DEFAULT_TTS_VOICE = os.getenv("KIOSK_CORE_TTS_VOICE")
DEFAULT_TTS_LANGUAGE = os.getenv("KIOSK_CORE_TTS_LANGUAGE", "English")
# ASR language hint sent with every transcription request. Left unset, Whisper
# auto-detects per 5-second chunk and regularly mis-fires on short kiosk
# utterances — measured over 25 real session chunks it produced Spanish output
# ("¿No puedo abrir el restaurante...") and one runaway repetition loop.
# Forcing "en" cut word error rate from 11.0% to 7.0% on CPU, GPU and NPU
# alike. Set to an empty string to restore auto-detection.
DEFAULT_ASR_LANGUAGE = os.getenv("KIOSK_CORE_ASR_LANGUAGE", "en") or None
DEFAULT_TTS_INSTRUCTIONS = os.getenv("KIOSK_CORE_TTS_INSTRUCTIONS")
DEFAULT_SAMPLE_RATE = int(os.getenv("KIOSK_CORE_SAMPLE_RATE", "16000"))

# Audio capture source override.
# When HOST_MIC=true the backend captures audio directly from the host machine's
# microphone; otherwise the browser captures audio and streams it to the backend.
# This lets the same build work both locally (host mic) and against a
# remote/headless kiosk-core (browser mic) without auto-detection surprises.
HOST_MIC = os.getenv("HOST_MIC", "false").lower() not in ("false", "0", "no")

# ---------------------------------------------------------------------------
# TTS segment silence trimming
# ---------------------------------------------------------------------------
# The response is split at clause boundaries (",", ".", "!") so synthesis can
# start streaming after the first fragment — measured time-to-first-audio is
# 387 ms this way versus 1769 ms when splitting on sentences only, so the split
# is worth keeping. The cost is that every fragment arrives with its own
# leading/trailing silence baked in by the TTS model: across 10 real production
# segments that was 1.76 s of dead air (8% of the reply), heard as a long pause
# at every comma.
#
# Trimming each segment back to a deliberate pad turns that accidental pause
# into a controlled one. The pad is intentionally non-zero — clauses should
# breathe — but short enough to sound continuous.
DEFAULT_TTS_TRIM_ENABLED = os.getenv("KIOSK_CORE_TTS_TRIM_ENABLED", "true").lower() not in ("false", "0", "no")

# When list_products is called with no category, return a per-category summary
# instead of every product. The catalogue is 26 items: reciting it costs ~19 s
# of LLM generation and ~40 s of speech, and no kiosk customer listens to a
# 26-item list. Prompt rules alone did not stop the model reciting it.
# Set false to restore the full listing.
DEFAULT_LIST_PRODUCTS_SUMMARY = os.getenv(
    "KIOSK_CORE_LIST_PRODUCTS_SUMMARY", "true"
).lower() not in ("false", "0", "no")
# Silence kept before speech starts, in every segment.
DEFAULT_TTS_LEAD_PAD_MS = float(os.getenv("KIOSK_CORE_TTS_LEAD_PAD_MS", "20"))
# Trailing silence for a fragment ending mid-sentence (",", ":", ";").
DEFAULT_TTS_CLAUSE_PAD_MS = float(os.getenv("KIOSK_CORE_TTS_CLAUSE_PAD_MS", "60"))
# Trailing silence for a fragment ending a sentence (".", "!", "?").
DEFAULT_TTS_SENTENCE_PAD_MS = float(os.getenv("KIOSK_CORE_TTS_SENTENCE_PAD_MS", "150"))
# Amplitude below this fraction of the segment peak counts as silence.
DEFAULT_TTS_SILENCE_FLOOR = float(os.getenv("KIOSK_CORE_TTS_SILENCE_FLOOR", "0.02"))

# Metrics collector – base URL of the standalone metrics-collector container.
# Within Docker the service is reachable as http://metrics-collector:9000.
METRICS_COLLECTOR_URL = os.getenv(
    "KIOSK_CORE_METRICS_URL",
    "http://metrics-collector:9000",
)
DEFAULT_CHUNK_SECONDS = float(os.getenv("KIOSK_CORE_CHUNK_SECONDS", "5.0"))
DEFAULT_SILENCE_TIMEOUT_SECONDS = float(os.getenv("KIOSK_CORE_SILENCE_TIMEOUT_SECONDS", "1.5"))
DEFAULT_MAX_SESSION_SECONDS = float(os.getenv("KIOSK_CORE_MAX_SESSION_SECONDS", "20.0"))
DEFAULT_SILENCE_THRESHOLD = int(os.getenv("KIOSK_CORE_SILENCE_THRESHOLD", "900"))
DEFAULT_BLOCK_DURATION_SECONDS = float(os.getenv("KIOSK_CORE_BLOCK_DURATION_SECONDS", "0.1"))
DEFAULT_PREROLL_SECONDS = float(os.getenv("KIOSK_CORE_PREROLL_SECONDS", "0.3"))
DEFAULT_HTTP_TIMEOUT_SECONDS = float(os.getenv("KIOSK_CORE_HTTP_TIMEOUT_SECONDS", "300.0"))

# Wake-word activation (openwakeword)
WAKEWORD_ENABLED = os.getenv("KIOSK_CORE_WAKEWORD_ENABLED", "false").lower() not in ("false", "0", "no")
DEFAULT_WAKEWORD_MODEL = os.getenv("KIOSK_CORE_WAKEWORD_MODEL", "hey jarvis")
DEFAULT_WAKEWORD_THRESHOLD = float(os.getenv("KIOSK_CORE_WAKEWORD_THRESHOLD", "0.5"))
DEFAULT_WAKEWORD_VAD_THRESHOLD = float(os.getenv("KIOSK_CORE_WAKEWORD_VAD_THRESHOLD", "0.4"))
DEFAULT_WAKEWORD_PATIENCE_FRAMES = int(os.getenv("KIOSK_CORE_WAKEWORD_PATIENCE_FRAMES", "2"))
DEFAULT_WAKEWORD_TIMEOUT_SECONDS = float(os.getenv("KIOSK_CORE_WAKEWORD_TIMEOUT_SECONDS", "0"))
DEFAULT_WAKEWORD_INFERENCE_FRAMEWORK = os.getenv("KIOSK_CORE_WAKEWORD_INFERENCE_FRAMEWORK", "onnx")

# Speaker diarization — master switch and semantic fallback sensitivity.
# Set KIOSK_CORE_DIARIZATION_ENABLED=false to revert to flat-text behavior
# (no speaker filtering; all segments forwarded as-is).
DEFAULT_DIARIZATION_ENABLED = os.getenv("KIOSK_CORE_DIARIZATION_ENABLED", "true").lower() not in ("false", "0", "no")
# Minimum domain-keyword overlap ratio to accept a fallback segment when the
# primary customer is silent for an entire chunk.
DEFAULT_SEMANTIC_FALLBACK_THRESHOLD = float(os.getenv("KIOSK_CORE_SEMANTIC_FALLBACK_THRESHOLD", "0.10"))
# When the audio-analyzer holds an enrolled reference voice it tags every
# segment with is_primary. If it marks them all non-primary that is a positive
# rejection and the chunk is dropped, so a bystander cannot inject orders.
# Set KIOSK_CORE_SPEAKER_STRICT_DROP=false to fall back to the first-speaker /
# semantic heuristics instead — an escape hatch for when voice enrollment is
# mistuned and starts rejecting the real customer.
DEFAULT_SPEAKER_STRICT_DROP = os.getenv("KIOSK_CORE_SPEAKER_STRICT_DROP", "true").lower() not in ("false", "0", "no")

# ── Ordering & Agent feature ─────────────────────────────────────────────────
# Set KIOSK_CORE_ORDERING_ENABLED=false to disable the ordering/agent feature
# and keep the legacy RAG-only Q&A flow.
ORDERING_ENABLED = os.getenv("KIOSK_CORE_ORDERING_ENABLED", "true").lower() not in ("false", "0", "no")

# Single shared kiosk identity used for ordering when no per-user login is
# wired into the request (this kiosk currently serves one customer at a time).
DEFAULT_ORDERING_USER_ID = os.getenv("KIOSK_CORE_DEFAULT_USER_ID", "kiosk-user")

# RAG-service agent chat endpoint (for ordering turns).
DEFAULT_AGENT_URL = os.getenv(
    "KIOSK_CORE_AGENT_URL",
    "http://127.0.0.1:8020/api/v1/agent/chat",
)

# Consume the agent's streaming endpoint so complete sentences reach TTS as
# they are generated instead of after the whole turn.
#
# Measured: the first sentence of a reply exists at ~700 ms while the full
# reply takes 1.3-4.0 s, so time-to-first-audio drops from ~5.0 s to ~2.8 s.
#
# OFF by default, and it must stay paired with AGENT_STREAM_SENTENCES on
# rag-service: the agent only releases sentences that provably cannot be
# rewritten by a later guard, and this client still re-validates the
# authoritative reply before speaking any remainder. Set to false to fall back
# to the buffered endpoint without a rebuild.
AGENT_STREAM_ENABLED = os.getenv(
    "KIOSK_CORE_AGENT_STREAM_ENABLED", "false"
).lower() in ("true", "1", "yes")

# Derived from DEFAULT_AGENT_URL so both point at the same service.
DEFAULT_AGENT_STREAM_URL = os.getenv(
    "KIOSK_CORE_AGENT_STREAM_URL",
    DEFAULT_AGENT_URL.rstrip("/") + "/stream",
)

# SQLite database file path (ordering domain).
KIOSK_DB_PATH = os.getenv("KIOSK_CORE_DB_PATH", "./kiosk.db")

# YAML seed files for product catalogue and upsell rules.
PRODUCTS_YAML_PATH = os.getenv(
    "KIOSK_CORE_PRODUCTS_YAML",
    "./configs/ordering/products.yaml",
)
UPSELL_RULES_YAML_PATH = os.getenv(
    "KIOSK_CORE_UPSELL_RULES_YAML",
    "./configs/ordering/upsell_rules.yaml",
)
# Maximum number of upsell suggestions attached to a place_order/update_order
# result.  Every extra suggestion the agent has to speak costs ~8 output tokens
# (~350 ms of LLM decode on Panther Lake iGPU), so this directly trades upsell
# breadth against spoken-reply latency.  1 keeps the turn inside the 3-4 s SLA.
UPSELL_MAX_SUGGESTIONS = int(os.getenv("KIOSK_CORE_UPSELL_MAX_SUGGESTIONS", "1"))

# ── Identity / biometric authentication feature ──────────────────────────────
# Master switch for the multimodal (face + voice) identity subsystem.  When
# false, kiosk-core does not mount the identity router, does not construct the
# IdentityClient, and the standalone identity-service container is never called.
# Set KIOSK_CORE_IDENTITY_ENABLED=true to turn the feature on (the
# identity-service container must also be started, e.g. via the `identity`
# compose profile).
IDENTITY_ENABLED = os.getenv("KIOSK_CORE_IDENTITY_ENABLED", "false").lower() not in ("false", "0", "no")

# Base URL of the standalone identity-service.  Within Docker the service is
# reachable as http://identity-service:8013.
IDENTITY_SERVICE_URL = os.getenv(
    "KIOSK_CORE_IDENTITY_URL",
    "http://127.0.0.1:8013",
)

# ---------------------------------------------------------------------------
# Queue-service integration (dynamic peak-hour menu)
# ---------------------------------------------------------------------------
# When enabled, the queue-service exposes a queue count that kiosk-core can
# query (future server-side menu filtering).  The UI also polls this directly
# via /queue-svc/api/v1/queue/count proxied through nginx.
QUEUE_SERVICE_ENABLED = os.getenv("KIOSK_CORE_QUEUE_SERVICE_ENABLED", "true").lower() not in ("false", "0", "no")

QUEUE_SERVICE_URL = os.getenv(
    "KIOSK_CORE_QUEUE_SERVICE_URL",
    "http://127.0.0.1:8090",
)
