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
DEFAULT_TTS_INSTRUCTIONS = os.getenv("KIOSK_CORE_TTS_INSTRUCTIONS")
DEFAULT_SAMPLE_RATE = int(os.getenv("KIOSK_CORE_SAMPLE_RATE", "16000"))

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

# Speaker diarization — master switch and semantic fallback sensitivity.
# Set KIOSK_CORE_DIARIZATION_ENABLED=false to revert to flat-text behavior
# (no speaker filtering; all segments forwarded as-is).
DEFAULT_DIARIZATION_ENABLED = os.getenv("KIOSK_CORE_DIARIZATION_ENABLED", "true").lower() not in ("false", "0", "no")
# Minimum domain-keyword overlap ratio to accept a fallback segment when the
# primary customer is silent for an entire chunk.
DEFAULT_SEMANTIC_FALLBACK_THRESHOLD = float(os.getenv("KIOSK_CORE_SEMANTIC_FALLBACK_THRESHOLD", "0.10"))

# ── Ordering & Agent feature ─────────────────────────────────────────────────
# Set KIOSK_CORE_ORDERING_ENABLED=false to disable the ordering/agent feature
# and keep the legacy RAG-only Q&A flow.
ORDERING_ENABLED = os.getenv("KIOSK_CORE_ORDERING_ENABLED", "true").lower() not in ("false", "0", "no")

# RAG-service agent chat endpoint (for ordering turns).
DEFAULT_AGENT_URL = os.getenv(
    "KIOSK_CORE_AGENT_URL",
    "http://127.0.0.1:8020/api/v1/agent/chat",
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
