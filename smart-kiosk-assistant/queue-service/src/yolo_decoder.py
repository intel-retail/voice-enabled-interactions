# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""YOLO26 detection decoder for queue-service.

Executed by a ``gvapython`` element placed immediately after ``gvainference``
and before ``gvatrack``. It converts the raw YOLO26 output tensor into
``GstVideoRegionOfInterestMeta`` person regions, so that ``gvatrack``,
``gvawatermark`` and ``QueueCounter`` all behave exactly as they do with the
Intel person-only detector -- nothing downstream needs to know the detector
changed.

Why this exists instead of ``gvadetect``
----------------------------------------
``gvadetect`` post-processes detections with a named converter from the
model-proc JSON. The converters compiled into DLStreamer 2026.1.0 *and*
2026.2.0 top out at ``yolo_v8``. YOLO26's ``yolo_v10``-style output is
*accepted* by the ``yolo_v10``/``yolo_v11`` converter names -- no error, no
warning -- but parses to **zero detections**, while a genuinely unknown
converter name is rejected loudly. That silent-success failure mode is the
reason this decoder exists: the model is fine (raw OpenVINO scores the same
frame at 0.89 person confidence), only DLStreamer's parsing of it is not.

Owning the post-processing also removes the dependency entirely -- a future
DLStreamer upgrade cannot silently change how detections are decoded.

Output tensor layout
--------------------
YOLO26 is end-to-end / NMS-free: the network emits a fixed-size, already
deduplicated, confidence-sorted set of candidates::

    [1, 300, 6]  ->  300 rows of (x1, y1, x2, y2, score, class_id)

Coordinates are pixels in the 640x640 *model input* space, NOT normalized and
NOT in frame space. Because the detections are already NMS-free, this decoder
performs no IoU suppression -- doing so would be redundant work per frame.

Letterbox inversion (the subtle part)
-------------------------------------
``gvainference`` preprocesses with aspect-ratio-preserving resize and pads the
short axis **symmetrically (centered)**, not top-left. This was verified
against raw OpenVINO on an identical frame: a 1920x1080 source scales by
r=1/3 to 640x360, leaving 280 rows of padding, and the pipeline's boxes sat
exactly 140 px (=280/2) lower in y than an explicit top-left letterbox.

Getting this wrong does not fail loudly -- every box simply shifts, which
silently skews ROI membership and therefore the queue count. The inversion is
covered by unit tests for that reason.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_KEEP_ID = 0
_DEFAULT_KEEP_LABEL = "person"
_DEFAULT_THRESHOLD = 0.5
# Model input is square for YOLO26; read from config when it ever is not.
_DEFAULT_INPUT_SIZE = 640
_ROW_STRIDE = 6


class YoloDecoder:
    """gvapython callback decoding YOLO26 tensors into person regions."""

    def __init__(
        self,
        threshold: float | None = None,
        keep_id: int | None = None,
        keep_label: str | None = None,
        input_size: int | None = None,
    ) -> None:
        model = self._model_config()
        if threshold is None:
            threshold = model.get("threshold", _DEFAULT_THRESHOLD)
        if keep_id is None:
            raw = model.get("class_id")
            keep_id = _DEFAULT_KEEP_ID if raw is None else int(raw)
        if keep_label is None:
            keep_label = model.get("class_label") or _DEFAULT_KEEP_LABEL
        if input_size is None:
            input_size = int(model.get("input_size") or _DEFAULT_INPUT_SIZE)

        self._threshold = float(threshold)
        self._keep_id = int(keep_id)
        self._keep_label = str(keep_label)
        self._input_size = int(input_size)
        self._warned_empty = False

        logger.info(
            "YoloDecoder ready (threshold=%.2f, keep_id=%d, keep_label=%s, input=%d)",
            self._threshold, self._keep_id, self._keep_label, self._input_size,
        )

    # ── configuration ────────────────────────────────────────────────────────

    @staticmethod
    def _model_config() -> dict:
        try:
            from config_loader import config

            model = getattr(config, "model", None)
            if model is None:
                return {}
            return vars(model) if hasattr(model, "__dict__") else dict(model)
        except Exception:  # noqa: BLE001 - config is optional for unit tests
            return {}

    # ── decoding ─────────────────────────────────────────────────────────────

    def decode(
        self, raw: np.ndarray, frame_width: int, frame_height: int
    ) -> list[tuple[int, int, int, int, float]]:
        """Decode a raw YOLO26 tensor into frame-space person boxes.

        Split out from :meth:`process_frame` so the coordinate maths can be
        unit-tested without GStreamer.

        Args:
            raw: Flat or shaped tensor holding rows of
                ``(x1, y1, x2, y2, score, class_id)`` in model-input pixels.
            frame_width: Source frame width in pixels.
            frame_height: Source frame height in pixels.

        Returns:
            ``(x, y, w, h, confidence)`` tuples in frame pixel coordinates,
            clamped to the frame and with zero-area boxes removed.
        """
        if raw is None or frame_width <= 0 or frame_height <= 0:
            return []

        arr = np.asarray(raw, dtype=np.float32).reshape(-1)
        if arr.size < _ROW_STRIDE:
            return []
        # Tolerate a trailing partial row rather than raising on an unexpected
        # tensor size; a malformed frame must not kill the pipeline.
        usable = (arr.size // _ROW_STRIDE) * _ROW_STRIDE
        rows = arr[:usable].reshape(-1, _ROW_STRIDE)

        keep = (rows[:, 4] >= self._threshold) & (
            np.rint(rows[:, 5]).astype(np.int32) == self._keep_id
        )
        rows = rows[keep]
        if rows.size == 0:
            return []

        size = float(self._input_size)
        scale = min(size / frame_width, size / frame_height)
        if scale <= 0:
            return []
        # Symmetric (centered) padding -- see module docstring.
        pad_x = (size - frame_width * scale) / 2.0
        pad_y = (size - frame_height * scale) / 2.0

        x1 = (rows[:, 0] - pad_x) / scale
        y1 = (rows[:, 1] - pad_y) / scale
        x2 = (rows[:, 2] - pad_x) / scale
        y2 = (rows[:, 3] - pad_y) / scale

        x1 = np.clip(x1, 0, frame_width)
        y1 = np.clip(y1, 0, frame_height)
        x2 = np.clip(x2, 0, frame_width)
        y2 = np.clip(y2, 0, frame_height)

        out: list[tuple[int, int, int, int, float]] = []
        for bx1, by1, bx2, by2, score in zip(
            x1, y1, x2, y2, rows[:, 4], strict=False
        ):
            w = int(round(float(bx2 - bx1)))
            h = int(round(float(by2 - by1)))
            if w <= 0 or h <= 0:
                # A box clipped entirely outside the frame carries no
                # information and would confuse the tracker's IoU matching.
                continue
            out.append((int(round(float(bx1))), int(round(float(by1))), w, h, float(score)))
        return out

    # ── gvapython entry point ────────────────────────────────────────────────

    def process_frame(self, frame) -> bool:
        """Attach decoded person regions to the frame metadata."""
        try:
            tensors = list(frame.tensors())
        except Exception:  # noqa: BLE001 - never break the pipeline
            logger.exception("YoloDecoder could not read frame tensors")
            return True

        if not tensors:
            if not self._warned_empty:
                self._warned_empty = True
                logger.warning(
                    "YoloDecoder received a frame with no inference tensor; "
                    "is gvainference upstream of this element?"
                )
            return True

        try:
            data = tensors[0].data()
            info = frame.video_info()
            boxes = self.decode(data, info.width, info.height)
            for x, y, w, h, score in boxes:
                frame.add_region(x, y, w, h, self._keep_label, score)
        except Exception:  # noqa: BLE001 - never break the pipeline
            logger.exception("YoloDecoder failed to decode a frame")
        return True
