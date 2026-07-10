"""Thread-safe ring buffer that stores the latest JPEG frame.

``put()`` is called from the GStreamer appsink callback (GLib thread).
``get_jpeg()`` is polled by the MJPEG streaming endpoint (asyncio thread).
A plain threading.Lock avoids any asyncio/GLib cross-loop interaction.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

_lock = threading.Lock()
_jpeg: bytes | None = None
_quality = 80


def configure(jpeg_quality: int = 80) -> None:
    """Set JPEG encoding quality (0-100, higher = better)."""
    global _quality  # noqa: PLW0603
    _quality = max(1, min(100, jpeg_quality))


def put(bgr_frame: np.ndarray) -> None:
    """Encode a BGR frame to JPEG and store it (overwrites previous frame).

    Called from the GStreamer thread — must never block for long.
    """
    ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, _quality])
    if not ok:
        return
    data = buf.tobytes()
    with _lock:
        global _jpeg  # noqa: PLW0603
        _jpeg = data


def get_jpeg() -> bytes | None:
    """Return the latest JPEG bytes, or ``None`` if no frame has been stored."""
    with _lock:
        return _jpeg
