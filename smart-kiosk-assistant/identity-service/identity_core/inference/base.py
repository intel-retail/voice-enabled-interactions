"""Inference contracts for the identity-service.

Defines the Strategy interfaces that every embedding backend must implement so
that the service/repository layers never depend on a concrete ML framework.
Both engines return **L2-normalized ``float32``** vectors, ready to be written
to (or searched against) the FAISS ``IndexFlatIP`` stores.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    """A single detected face in normalized [0, 1] image coordinates."""

    confidence: float
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return ``vector`` as a 1-D L2-normalized ``float32`` array.

    Args:
        vector: Raw embedding of any shape; flattened before normalization.

    Returns:
        A contiguous ``float32`` unit vector (zero vectors are returned
        unchanged to avoid division-by-zero).
    """
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flat))
    if norm > 0.0:
        flat = flat / norm
    return np.ascontiguousarray(flat, dtype=np.float32)


class FaceEmbeddingEngine(abc.ABC):
    """Strategy interface: image (BGR) → face identity embedding."""

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Dimensionality of the produced embedding."""

    @abc.abstractmethod
    def embed(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Detect the most prominent face and return its embedding.

        Args:
            image_bgr: ``HxWx3`` BGR image (OpenCV convention).

        Returns:
            An L2-normalized ``float32`` embedding, or ``None`` when no face
            clears the detector confidence threshold.
        """


class VoiceEmbeddingEngine(abc.ABC):
    """Strategy interface: mono waveform → speaker embedding."""

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Dimensionality of the produced embedding."""

    @abc.abstractmethod
    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a speaker embedding for a mono waveform.

        Args:
            waveform: 1-D ``float32`` PCM samples in ``[-1, 1]``.
            sample_rate: Sample rate of ``waveform`` in Hz.

        Returns:
            An L2-normalized ``float32`` embedding.
        """
