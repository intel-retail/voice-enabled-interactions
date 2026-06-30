"""Typed configuration for the identity-service.

Loads defaults from ``identity_config.yaml`` and overlays environment-variable
overrides.  Business code references the typed ``Settings`` object — never raw
``os.getenv`` calls or ``dict["key"]`` access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Path to the YAML config (mounted read-only in the container).
_CONFIG_YAML = os.getenv(
    "IDENTITY_CONFIG_YAML",
    "/app/configs/identity/identity_config.yaml",
)
_PROMPTS_YAML = os.getenv(
    "IDENTITY_PROMPTS_YAML",
    "/app/configs/identity/verification_prompts.yaml",
)


def _as_bool(value: str | bool | None, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() not in ("false", "0", "no", "")


@dataclass(frozen=True)
class Settings:
    """Resolved identity-service settings (YAML defaults + env overrides)."""

    bootstrap_on_start: bool = True

    # Storage
    db_path: str = "/app/data/kiosk.db"
    faiss_dir: str = "/app/data/identity"
    face_index_file: str = "face_index.bin"
    voice_index_file: str = "voice_index.bin"
    face_embedding_dim: int = 256
    voice_embedding_dim: int = 192

    # Thresholds / fusion
    fusion_face_weight: float = 0.6
    fusion_voice_weight: float = 0.4
    combined_threshold: float = 0.78
    face_threshold: float = 0.80
    voice_threshold: float = 0.75

    # OpenVINO models
    models_dir: str = "/app/models"
    identity_subdir: str = "identity"
    precision: str = "FP16"
    device: str = "GPU"
    face_detection_model: str = "face-detection-retail-0005"
    face_reid_model: str = "face-reidentification-retail-0095"
    voice_embedding_model: str = "ecapa-tdnn-voice"
    voice_backend: str = "openvino"  # openvino | speechbrain (Strategy switch)
    video_frame_sample_rate: int = 10
    face_detection_min_confidence: float = 0.7

    # Voice feature extraction (used only by waveform-input backends).
    voice_sample_rate: int = 16000

    # Network
    host: str = "0.0.0.0"
    port: int = 8013

    # Bootstrap profiles + challenge prompts (loaded from YAML).
    profiles: list[dict] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)

    @property
    def face_index_path(self) -> Path:
        return Path(self.faiss_dir) / self.face_index_file

    @property
    def voice_index_path(self) -> Path:
        return Path(self.faiss_dir) / self.voice_index_file

    @property
    def _identity_models_dir(self) -> Path:
        return Path(self.models_dir) / self.identity_subdir

    @property
    def face_detection_xml(self) -> Path:
        """IR graph for the face detector (OMZ layout: <model>/<PRECISION>/<model>.xml)."""
        return (
            self._identity_models_dir
            / self.face_detection_model
            / self.precision
            / f"{self.face_detection_model}.xml"
        )

    @property
    def face_reid_xml(self) -> Path:
        """IR graph for the 256-d face re-identification model."""
        return (
            self._identity_models_dir
            / self.face_reid_model
            / self.precision
            / f"{self.face_reid_model}.xml"
        )

    @property
    def voice_xml(self) -> Path:
        """IR graph for the converted ECAPA-TDNN voice embedding model."""
        return (
            self._identity_models_dir
            / self.voice_embedding_model
            / "openvino"
            / f"{self.voice_embedding_model}.xml"
        )


def _load_yaml(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as fh:
        return yaml.safe_load(fh) or {}


def load_settings() -> Settings:
    """Build the Settings object from YAML defaults overlaid with env vars."""
    cfg = _load_yaml(_CONFIG_YAML)
    prompts_cfg = _load_yaml(_PROMPTS_YAML)

    def _get(key: str, default):
        return cfg.get(key, default)

    defaults = Settings()
    settings = Settings(
        bootstrap_on_start=_as_bool(
            os.getenv("BOOTSTRAP_ON_START"), _as_bool(_get("bootstrap_on_start", True), True)
        ),
        db_path=os.getenv("IDENTITY_DB_PATH", _get("db_path", defaults.db_path)),
        faiss_dir=os.getenv("IDENTITY_FAISS_DIR", _get("faiss_dir", defaults.faiss_dir)),
        face_index_file=_get("face_index_file", defaults.face_index_file),
        voice_index_file=_get("voice_index_file", defaults.voice_index_file),
        face_embedding_dim=int(_get("face_embedding_dim", defaults.face_embedding_dim)),
        voice_embedding_dim=int(_get("voice_embedding_dim", defaults.voice_embedding_dim)),
        fusion_face_weight=float(_get("fusion_face_weight", defaults.fusion_face_weight)),
        fusion_voice_weight=float(_get("fusion_voice_weight", defaults.fusion_voice_weight)),
        combined_threshold=float(
            os.getenv("IDENTITY_COMBINED_THRESHOLD", _get("combined_threshold", defaults.combined_threshold))
        ),
        face_threshold=float(_get("face_threshold", defaults.face_threshold)),
        voice_threshold=float(_get("voice_threshold", defaults.voice_threshold)),
        models_dir=os.getenv("IDENTITY_MODELS_DIR", _get("models_dir", defaults.models_dir)),
        identity_subdir=_get("identity_subdir", defaults.identity_subdir),
        precision=os.getenv("IDENTITY_PRECISION", _get("precision", defaults.precision)).upper(),
        device=os.getenv("IDENTITY_DEVICE", _get("device", defaults.device)).upper(),
        face_detection_model=_get("face_detection_model", defaults.face_detection_model),
        face_reid_model=_get("face_reid_model", defaults.face_reid_model),
        voice_embedding_model=_get("voice_embedding_model", defaults.voice_embedding_model),
        voice_backend=os.getenv("IDENTITY_VOICE_BACKEND", _get("voice_backend", defaults.voice_backend)).lower(),
        video_frame_sample_rate=int(_get("video_frame_sample_rate", defaults.video_frame_sample_rate)),
        face_detection_min_confidence=float(
            _get("face_detection_min_confidence", defaults.face_detection_min_confidence)
        ),
        voice_sample_rate=int(_get("voice_sample_rate", defaults.voice_sample_rate)),
        host=os.getenv("IDENTITY_HOST", defaults.host),
        port=int(os.getenv("IDENTITY_PORT", str(defaults.port))),
        profiles=list(_get("profiles", []) or []),
        prompts=list(prompts_cfg.get("prompts", []) or []),
    )
    return settings
