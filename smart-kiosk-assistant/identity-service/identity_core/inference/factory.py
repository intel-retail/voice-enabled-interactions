"""Factory that assembles inference engines from typed ``Settings``.

Centralizes backend selection (Strategy) and model-file validation so the
service layer stays free of OpenVINO/import concerns.  If any required model
file is missing, the corresponding engine is reported as unavailable and the
service keeps running with ``inference_ready=False`` (verify/register stay
gated) rather than crashing at startup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from identity_core.config import Settings
from identity_core.inference.base import FaceEmbeddingEngine, VoiceEmbeddingEngine

logger = logging.getLogger(__name__)


@dataclass
class InferenceEngines:
    """Container for the built engines (either may be ``None`` if unavailable)."""

    face: FaceEmbeddingEngine | None = None
    voice: VoiceEmbeddingEngine | None = None

    @property
    def ready(self) -> bool:
        """True only when both modalities are loaded (auth requires both)."""
        return self.face is not None and self.voice is not None


def _missing(*paths: Path) -> list[Path]:
    return [p for p in paths if not p.exists()]


def build_face_engine(settings: Settings) -> FaceEmbeddingEngine | None:
    """Build the face engine, or ``None`` if models/runtime are unavailable."""
    missing = _missing(settings.face_detection_xml, settings.face_reid_xml)
    if missing:
        logger.warning(
            "[IDENTITY] Face engine disabled — missing model files: %s",
            ", ".join(str(p) for p in missing),
        )
        return None
    try:
        from identity_core.inference.openvino_face import OpenVinoFaceEngine

        return OpenVinoFaceEngine(
            detection_xml=settings.face_detection_xml,
            reid_xml=settings.face_reid_xml,
            device=settings.device,
            min_confidence=settings.face_detection_min_confidence,
            embedding_dim=settings.face_embedding_dim,
        )
    except Exception:  # noqa: BLE001 — degrade gracefully on any load failure
        logger.exception("[IDENTITY] Failed to initialize face engine")
        return None


def build_voice_engine(settings: Settings) -> VoiceEmbeddingEngine | None:
    """Build the voice engine for the configured backend, or ``None``."""
    backend = settings.voice_backend
    if backend != "openvino":
        logger.warning(
            "[IDENTITY] Voice backend '%s' not implemented; only 'openvino' is "
            "available in Phase 4. Voice engine disabled.",
            backend,
        )
        return None

    missing = _missing(settings.voice_xml)
    if missing:
        logger.warning(
            "[IDENTITY] Voice engine disabled — missing model file: %s",
            ", ".join(str(p) for p in missing),
        )
        return None
    try:
        from identity_core.inference.openvino_voice import OpenVinoVoiceEngine

        return OpenVinoVoiceEngine(
            voice_xml=settings.voice_xml,
            device=settings.device,
            embedding_dim=settings.voice_embedding_dim,
            target_sample_rate=settings.voice_sample_rate,
        )
    except Exception:  # noqa: BLE001 — degrade gracefully on any load failure
        logger.exception("[IDENTITY] Failed to initialize voice engine")
        return None


def build_engines(settings: Settings) -> InferenceEngines:
    """Build both engines; missing modalities degrade to ``None`` (not fatal)."""
    engines = InferenceEngines(
        face=build_face_engine(settings),
        voice=build_voice_engine(settings),
    )
    logger.info(
        "[IDENTITY] Inference engines built — face=%s voice=%s ready=%s",
        engines.face is not None,
        engines.voice is not None,
        engines.ready,
    )
    return engines
