# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Metadata-level person filtering for queue-service.

Executed by a ``gvapython`` element placed immediately after ``gvadetect`` and
before ``gvatrack``. Multi-class detectors (YOLOv8/YOLO11) emit detections for
every COCO class; this callback removes every region that is not the configured
person class *at the metadata level*, so that ``gvatrack`` only tracks persons,
``gvawatermark`` only draws persons and ``QueueCounter`` only ever receives
person detections.

A region is kept when its string label matches ``model.class_label`` *or* its
numeric class id matches ``model.class_id`` (defaults ``"person"``/``0``). The
class id is read defensively because gstgva exposes it as either ``label_id()``
(newer) or ``class_id()`` (older), which keeps the filter robust across
different YOLO/OpenVINO exports.

The Intel ``person-detection-retail-0013`` model is person-only, so this
element is not inserted for it (see ``pipeline.py``) and its behaviour is
unchanged.

DLStreamer has no built-in class-filter element; the documented pattern for
custom metadata post-processing is a ``gvapython`` module that mutates the
frame metadata via ``VideoFrame.remove_region``. The module avoids importing
GStreamer at import time so it stays unit-testable.
"""
import logging

logger = logging.getLogger(__name__)

# Defaults kept when none are configured (COCO person => label "person", id 0).
_DEFAULT_KEEP_LABEL = "person"
_DEFAULT_KEEP_ID = 0


class PersonFilter:
    """gvapython callback that drops every non-person detection region."""

    def __init__(
        self,
        keep_label: str | None = None,
        keep_id: int | None = None,
    ) -> None:
        model = self._model_config()
        if keep_label is None:
            keep_label = model.get("class_label") or _DEFAULT_KEEP_LABEL
        if keep_id is None:
            raw_id = model.get("class_id")
            keep_id = _DEFAULT_KEEP_ID if raw_id is None else self._coerce_int(raw_id)
        self._keep_label = keep_label
        self._keep_id = keep_id

        logger.info(
            "PersonFilter ready (keep_label=%s, keep_id=%s)",
            self._keep_label, self._keep_id,
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
        except Exception:  # noqa: BLE001 - config optional for unit tests
            return {}

    @staticmethod
    def _coerce_int(value):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    # ── gvapython entry point ────────────────────────────────────────────────

    def process_frame(self, frame) -> bool:
        """Remove every region whose label is not the kept class.

        Returns ``True`` so the buffer continues downstream (gvapython
        contract). Failures are non-fatal -- on error the frame passes through
        unmodified rather than dropping the stream.
        """
        for region in self._non_person_regions(frame):
            try:
                frame.remove_region(region)
            except Exception:  # noqa: BLE001 - never break the stream on metadata edits
                logger.debug("Could not remove region; leaving it in place")
        return True

    # ── metadata helpers (defensive against gstgva version differences) ──────

    def _non_person_regions(self, frame) -> list:
        try:
            regions = list(frame.regions())
        except Exception:  # noqa: BLE001
            return []
        return [region for region in regions if not self._is_person(region)]

    def _is_person(self, region) -> bool:
        """Accept a region matching the kept class by label OR class id.

        Robust across detector exports: matches the string label when the
        model-proc supplies one, and also the numeric class id (``label_id()``
        on newer gstgva, ``class_id()`` on older builds). A region is kept when
        either matches; ambiguous regions (no usable label/id) are kept to
        avoid silently dropping detections.
        """
        label = self._region_label(region)
        class_id = self._region_class_id(region)
        if label is None and class_id is None:
            return True
        if label is not None and label == self._keep_label:
            return True
        if class_id is not None and self._keep_id is not None and class_id == self._keep_id:
            return True
        return False

    @staticmethod
    def _region_label(region):
        try:
            return region.label()
        except Exception:  # noqa: BLE001 - label optional on some exports
            return None

    @staticmethod
    def _region_class_id(region):
        # gstgva exposes either label_id() (newer) or class_id() (older).
        for attr in ("label_id", "class_id"):
            method = getattr(region, attr, None)
            if method is None:
                continue
            try:
                return int(method())
            except Exception:  # noqa: BLE001 - try the next accessor
                continue
        return None
