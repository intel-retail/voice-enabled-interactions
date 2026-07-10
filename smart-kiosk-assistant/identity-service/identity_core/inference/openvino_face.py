"""OpenVINO face embedding engine.

Two-stage pipeline using Intel Open Model Zoo IR models:

1. ``face-detection-retail-0005`` — SSD detector, input ``1x3x300x300`` BGR,
   output ``1x1xNx7`` rows ``[image_id, label, conf, x_min, y_min, x_max, y_max]``
   in normalized [0, 1] coordinates.
2. ``face-reidentification-retail-0095`` — input ``1x3x128x128`` BGR, output a
   ``256``-d descriptor (``1x256x1x1``).

The most confident face above the configured threshold is cropped and embedded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from identity_core.inference.base import FaceDetection, FaceEmbeddingEngine, l2_normalize

logger = logging.getLogger(__name__)

_DET_INPUT = (300, 300)   # (W, H) for face-detection-retail-0005
_REID_INPUT = (128, 128)  # (W, H) for face-reidentification-retail-0095


class OpenVinoFaceEngine(FaceEmbeddingEngine):
    """Face detection + re-identification via OpenVINO."""

    def __init__(
        self,
        detection_xml: Path,
        reid_xml: Path,
        device: str,
        min_confidence: float,
        embedding_dim: int,
    ) -> None:
        """Compile the detector and re-id models onto ``device``.

        Args:
            detection_xml: Path to the face-detection IR ``.xml``.
            reid_xml: Path to the face-reidentification IR ``.xml``.
            device: OpenVINO device string (uppercase: ``CPU``/``GPU``/``NPU``).
            min_confidence: Minimum detector confidence to accept a face.
            embedding_dim: Expected embedding dimensionality (for validation).
        """
        self._min_confidence = float(min_confidence)
        self._dim = int(embedding_dim)
        core = ov.Core()

        logger.info("[IDENTITY] Loading face detector %s on %s", detection_xml.name, device)
        det_model = core.read_model(detection_xml)
        self._det = core.compile_model(det_model, device)
        self._det_out = self._det.output(0)

        logger.info("[IDENTITY] Loading face re-id %s on %s", reid_xml.name, device)
        reid_model = core.read_model(reid_xml)
        self._reid = core.compile_model(reid_model, device)
        self._reid_out = self._reid.output(0)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, image_bgr: np.ndarray) -> np.ndarray | None:
        detection = self._detect_best(image_bgr)
        if detection is None:
            return None
        crop = self._crop(image_bgr, detection)
        if crop.size == 0:
            return None
        blob = self._preprocess(crop, _REID_INPUT)
        raw = self._reid(blob)[self._reid_out]
        return l2_normalize(raw)

    # ── internals ─────────────────────────────────────────────────────────────

    def _detect_best(self, image_bgr: np.ndarray) -> FaceDetection | None:
        """Run the detector and return the highest-confidence face, if any."""
        blob = self._preprocess(image_bgr, _DET_INPUT)
        raw = self._det(blob)[self._det_out]
        detections = raw.reshape(-1, 7)
        best: FaceDetection | None = None
        for row in detections:
            confidence = float(row[2])
            if confidence < self._min_confidence:
                continue
            if best is None or confidence > best.confidence:
                best = FaceDetection(
                    confidence=confidence,
                    x_min=float(row[3]),
                    y_min=float(row[4]),
                    x_max=float(row[5]),
                    y_max=float(row[6]),
                )
        return best

    @staticmethod
    def _crop(image_bgr: np.ndarray, det: FaceDetection) -> np.ndarray:
        """Crop the detected face, clamped to image bounds."""
        h, w = image_bgr.shape[:2]
        x1 = max(0, min(int(det.x_min * w), w - 1))
        y1 = max(0, min(int(det.y_min * h), h - 1))
        x2 = max(x1 + 1, min(int(det.x_max * w), w))
        y2 = max(y1 + 1, min(int(det.y_max * h), h))
        return image_bgr[y1:y2, x1:x2]

    @staticmethod
    def _preprocess(image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """Resize BGR image to ``size`` and return a ``1x3xHxW`` float32 blob."""
        resized = cv2.resize(image_bgr, size, interpolation=cv2.INTER_LINEAR)
        blob = resized.transpose(2, 0, 1)[np.newaxis, ...]
        return np.ascontiguousarray(blob, dtype=np.float32)
