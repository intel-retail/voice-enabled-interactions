"""OpenVINO ECAPA-TDNN voice embedding engine.

The model is a **full-pipeline** export (feature extraction + TDNN encoder +
attentive statistics pooling) converted from SpeechBrain
``speechbrain/spkrec-ecapa-voxceleb``.  It accepts a raw mono waveform tensor of
shape ``1xT`` (16 kHz, ``float32`` in ``[-1, 1]``) and emits a ``192``-d speaker
embedding, so no feature engineering is needed at runtime.

ECAPA-TDNN is text-independent, so the random challenge prompt serves purely as
an anti-replay nonce — the spoken words are never matched against it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import openvino as ov

from identity_core.inference.base import VoiceEmbeddingEngine, l2_normalize

logger = logging.getLogger(__name__)


class OpenVinoVoiceEngine(VoiceEmbeddingEngine):
    """ECAPA-TDNN speaker embedding via OpenVINO (raw-waveform input)."""

    def __init__(
        self,
        voice_xml: Path,
        device: str,
        embedding_dim: int,
        target_sample_rate: int,
    ) -> None:
        """Compile the voice IR onto ``device``.

        Args:
            voice_xml: Path to the converted ECAPA IR ``.xml``.
            device: OpenVINO device string (uppercase).
            embedding_dim: Expected embedding dimensionality (for validation).
            target_sample_rate: Sample rate the model was exported for (Hz).
        """
        self._dim = int(embedding_dim)
        self._target_sr = int(target_sample_rate)
        core = ov.Core()
        logger.info("[IDENTITY] Loading voice model %s on %s", voice_xml.name, device)
        model = core.read_model(voice_xml)
        self._model = core.compile_model(model, device)
        self._input = self._model.input(0)
        self._output = self._model.output(0)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        samples = self._prepare(waveform, sample_rate)
        blob = samples[np.newaxis, :]  # 1xT
        raw = self._model(blob)[self._output]
        return l2_normalize(raw)

    # ── internals ─────────────────────────────────────────────────────────────

    def _prepare(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Flatten to mono float32 and resample to the model's target rate."""
        samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if sample_rate != self._target_sr and samples.size > 0:
            samples = self._resample(samples, sample_rate, self._target_sr)
        return np.ascontiguousarray(samples, dtype=np.float32)

    @staticmethod
    def _resample(samples: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
        """Linear resample (sufficient for speaker-embedding fidelity)."""
        duration = samples.size / float(src_sr)
        dst_len = max(1, int(round(duration * dst_sr)))
        src_idx = np.linspace(0.0, samples.size - 1, num=samples.size, dtype=np.float64)
        dst_idx = np.linspace(0.0, samples.size - 1, num=dst_len, dtype=np.float64)
        return np.interp(dst_idx, src_idx, samples).astype(np.float32)
