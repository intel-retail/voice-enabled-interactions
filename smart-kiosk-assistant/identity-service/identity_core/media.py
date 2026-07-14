"""Media decoding helpers for the identity-service.

Converts the base64 payloads carried by ``VerifyRequest`` / ``RegisterRequest``
into the concrete inputs the OpenVINO engines expect:

* :func:`decode_image_bgr` -> ``HxWx3`` BGR ``uint8`` image for the face engine
  (``OpenVinoFaceEngine.embed`` handles detection/crop/resize internally).
* :func:`decode_wav` -> mono ``float32`` waveform in ``[-1, 1]`` plus its sample
  rate for the voice engine (``OpenVinoVoiceEngine.embed`` resamples to 16 kHz).

Both raise :class:`MediaDecodeError` on malformed input so the service layer can
turn a bad payload into a clean ``reason`` response instead of a 500.
"""

from __future__ import annotations

import base64
import binascii
import io
import wave

import cv2
import numpy as np


class MediaDecodeError(ValueError):
    """Raised when a base64 image or WAV payload cannot be decoded."""


# WAV sample width (bytes) -> (numpy dtype, full-scale divisor, zero offset).
# 8-bit PCM is unsigned and centred at 128; 16/32-bit PCM are signed.
_PCM_FORMATS: dict[int, tuple[type, float, float]] = {
    1: (np.uint8, 128.0, 128.0),
    2: (np.int16, 32768.0, 0.0),
    4: (np.int32, 2147483648.0, 0.0),
}


def _b64_to_bytes(base64_string: str) -> bytes:
    """Decode a base64 string to raw bytes, tolerating a data-URL prefix.

    Args:
        base64_string: Base64 payload, optionally prefixed with a browser
            ``data:<mime>;base64,`` header.

    Returns:
        The decoded raw bytes.

    Raises:
        MediaDecodeError: If the payload is empty or not valid base64.
    """
    if not base64_string:
        raise MediaDecodeError("Empty base64 payload.")
    # Strip an optional "data:<mime>;base64," prefix (browser camera captures).
    if base64_string.startswith("data:"):
        _, _, base64_string = base64_string.partition(",")
    try:
        return base64.b64decode(base64_string, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise MediaDecodeError(f"Invalid base64 payload: {exc}") from exc


def decode_image_bgr(base64_string: str) -> np.ndarray:
    """Decode a base64-encoded image into an ``HxWx3`` BGR array.

    Args:
        base64_string: Base64 (optionally data-URL-prefixed) JPEG/PNG frame.

    Returns:
        A contiguous ``HxWx3`` BGR ``uint8`` image (OpenCV convention), ready to
        pass straight to ``OpenVinoFaceEngine.embed``.

    Raises:
        MediaDecodeError: If the base64 is invalid or the bytes are not a
            decodable image.
    """
    raw = _b64_to_bytes(base64_string)
    buffer = np.frombuffer(raw, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise MediaDecodeError("Corrupt or unsupported image data.")
    return image


def decode_wav(base64_string: str) -> tuple[np.ndarray, int]:
    """Decode a base64-encoded WAV buffer into a mono ``float32`` waveform.

    Args:
        base64_string: Base64 (optionally data-URL-prefixed) PCM WAV buffer.

    Returns:
        ``(waveform, sample_rate)`` where ``waveform`` is a 1-D ``float32`` mono
        signal in ``[-1, 1]`` and ``sample_rate`` is in Hz, ready to pass to
        ``OpenVinoVoiceEngine.embed``.

    Raises:
        MediaDecodeError: If the base64 is invalid, the WAV is corrupt, or the
            sample width is unsupported.
    """
    raw = _b64_to_bytes(base64_string)
    try:
        with wave.open(io.BytesIO(raw), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise MediaDecodeError(f"Corrupt or unsupported WAV data: {exc}") from exc

    fmt = _PCM_FORMATS.get(sample_width)
    if fmt is None:
        raise MediaDecodeError(f"Unsupported WAV sample width: {sample_width} byte(s).")
    dtype, divisor, offset = fmt

    samples = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    samples = (samples - offset) / divisor  # normalize to [-1, 1]

    if channels > 1:
        # Downmix interleaved channels to mono by averaging. Trim any trailing
        # partial frame first so the reshape can never fail on truncated data.
        usable = (samples.size // channels) * channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)

    waveform = np.ascontiguousarray(samples, dtype=np.float32)
    return waveform, int(sample_rate)
