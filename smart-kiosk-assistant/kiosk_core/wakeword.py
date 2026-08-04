import logging
import io
import time
import threading
import wave
from dataclasses import dataclass
from queue import Empty, Queue

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


@dataclass
class WakeWordDetection:
    model: str
    detected_label: str
    score: float
    elapsed_seconds: float


class WakeWordListener:
    """Microphone wake-word listener powered by openwakeword."""

    def __init__(
        self,
        *,
        wakeword_model: str,
        threshold: float,
        vad_threshold: float,
        patience_frames: int,
        inference_framework: str,
    ) -> None:
        try:
            from openwakeword.model import Model
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "openwakeword is not installed. Install dependencies for kiosk-core and retry."
            ) from exc

        self.wakeword_model = wakeword_model
        self.threshold = float(threshold)
        self.vad_threshold = float(vad_threshold)
        self.patience_frames = int(patience_frames)

        try:
            self._model = self._build_model(Model, wakeword_model, inference_framework)
        except Exception as load_exc:  # noqa: BLE001
            raise RuntimeError(
                "Wake-word model assets are missing or failed to load. Automatic download is "
                "disabled because the openWakeWord pre-trained models are licensed for "
                "non-commercial use only (CC BY-NC-SA 4.0). Install the required models "
                "manually, or provide a local wakeword model path. "
                f"model='{wakeword_model}', framework='{inference_framework}', error={load_exc}"
            ) from load_exc

        self._target_slug = self._slug(wakeword_model)

    @staticmethod
    def _slug(text: str) -> str:
        return "".join(ch for ch in text.lower() if ch.isalnum())

    def _build_model(self, model_cls, wakeword_model: str, inference_framework: str):
        return model_cls(
            wakeword_models=[wakeword_model],
            vad_threshold=self.vad_threshold,
            inference_framework=inference_framework,
        )

    def _target_score(self, prediction: dict[str, float]) -> tuple[str, float]:
        # openwakeword can expose slightly different key names across models;
        # this keeps matching resilient to spaces/underscores/version suffixes.
        scored = []
        for label, score in prediction.items():
            label_slug = self._slug(label)
            if self._target_slug in label_slug or label_slug in self._target_slug:
                scored.append((label, float(score)))

        if scored:
            return max(scored, key=lambda item: item[1])

        # Single-model fallback.
        if prediction:
            best = max(prediction.items(), key=lambda item: float(item[1]))
            return str(best[0]), float(best[1])

        return self.wakeword_model, 0.0

    def listen(
        self,
        *,
        device: int | str | None,
        sample_rate: int,
        timeout_seconds: float,
    ) -> WakeWordDetection:
        if int(sample_rate) != 16000:
            raise ValueError("Wake-word detection requires sample_rate=16000")

        frame_size = 1280  # 80ms @ 16kHz
        q: Queue[np.ndarray] = Queue()
        consecutive_hits = 0
        start = time.monotonic()

        def _on_audio(indata, frames, callback_time, status):
            del frames, callback_time
            if status:
                logger.debug("Wake-word stream status: %s", status)
            q.put(indata[:, 0].copy())

        resolved_device = self._resolve_input_device(device)
        logger.info(
            "[WAKEWORD] Listening for '%s' on device=%s...",
            self.wakeword_model,
            resolved_device if resolved_device is not None else "default",
        )
        with sd.InputStream(
            samplerate=sample_rate,
            blocksize=frame_size,
            channels=1,
            dtype="int16",
            device=resolved_device,
            callback=_on_audio,
        ):
            while True:
                if timeout_seconds > 0 and (time.monotonic() - start) >= timeout_seconds:
                    raise TimeoutError(
                        f"Wake-word '{self.wakeword_model}' was not detected within {timeout_seconds:.1f}s"
                    )

                try:
                    frame = q.get(timeout=0.25)
                except Empty:
                    continue

                prediction = self._model.predict(frame)
                label, score = self._target_score(prediction)

                if score >= self.threshold:
                    consecutive_hits += 1
                else:
                    consecutive_hits = 0

                if consecutive_hits >= self.patience_frames:
                    elapsed = time.monotonic() - start
                    logger.info(
                        "[WAKEWORD] Detected '%s' (label=%s score=%.3f elapsed=%.2fs)",
                        self.wakeword_model,
                        label,
                        score,
                        elapsed,
                    )
                    return WakeWordDetection(
                        model=self.wakeword_model,
                        detected_label=label,
                        score=round(score, 4),
                        elapsed_seconds=round(elapsed, 3),
                    )

    @staticmethod
    def _resolve_input_device(device: int | str | None) -> int | str | None:
        """Return a valid input device for sounddevice InputStream.

        If the requested/default input device is invalid (often shown as -1 on
        some Windows setups), fall back to the first available input-capable
        device so wake-word mode can still start.
        """
        if device not in (None, "", -1, "-1"):
            return device

        try:
            default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else None
        except Exception:
            default_in = None

        if isinstance(default_in, int) and default_in >= 0:
            return default_in

        try:
            devices = sd.query_devices()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Unable to query audio input devices: {exc}") from exc

        for idx, dev in enumerate(devices):
            try:
                if int(dev.get("max_input_channels", 0)) > 0:
                    return idx
            except Exception:
                continue

        raise RuntimeError("No valid input microphone device was found on the host")


class BrowserWakeWordSession:
    """Stateful wake-word detector for browser-streamed WAV chunks."""

    def __init__(
        self,
        *,
        wakeword_model: str,
        threshold: float,
        vad_threshold: float,
        patience_frames: int,
        inference_framework: str,
        sample_rate: int,
    ) -> None:
        if int(sample_rate) != 16000:
            raise ValueError("Browser wake-word detection requires sample_rate=16000")

        self._listener = WakeWordListener(
            wakeword_model=wakeword_model,
            threshold=threshold,
            vad_threshold=vad_threshold,
            patience_frames=patience_frames,
            inference_framework=inference_framework,
        )
        self.sample_rate = int(sample_rate)
        self._frame_size = 1280
        self._consecutive_hits = 0
        self._detected = False
        self._last_score = 0.0
        self._last_label: str | None = None
        self._lock = threading.Lock()
        self._pcm_buffer = np.zeros(0, dtype=np.int16)

    def process_wav_chunk(self, wav_bytes: bytes) -> tuple[bool, str | None, float]:
        """Process one browser WAV chunk and return detection state."""
        with self._lock:
            try:
                if self._detected:
                    return True, self._last_label, self._last_score

                audio = self._decode_wav_to_pcm(wav_bytes)
                if audio.size == 0:
                    return False, self._last_label, self._last_score

                # Keep a rolling PCM buffer so inference always sees full
                # 80ms frames (1280 samples @ 16k). Partial tail samples from
                # chunk boundaries are carried into the next request.
                if self._pcm_buffer.size == 0:
                    self._pcm_buffer = audio
                else:
                    self._pcm_buffer = np.concatenate((self._pcm_buffer, audio))

                if self._pcm_buffer.size < self._frame_size:
                    return False, self._last_label, round(self._last_score, 4)

                n_complete = self._pcm_buffer.size // self._frame_size
                chunk_peak = 0.0
                for i in range(n_complete):
                    start = i * self._frame_size
                    frame = self._pcm_buffer[start : start + self._frame_size]
                    prediction = self._listener._model.predict(frame)
                    label, score = self._listener._target_score(prediction)
                    self._last_label = label
                    self._last_score = float(score)
                    if score > chunk_peak:
                        chunk_peak = float(score)

                    if score >= self._listener.threshold:
                        self._consecutive_hits += 1
                    else:
                        self._consecutive_hits = 0

                    if self._consecutive_hits >= self._listener.patience_frames:
                        self._detected = True
                        logger.info(
                            "[WAKEWORD] Browser stream detected '%s' (label=%s score=%.3f)",
                            self._listener.wakeword_model,
                            label,
                            score,
                        )
                        break

                # Diagnostic: surface near-miss peaks so the wake-word threshold
                # can be tuned. Silence stays quiet (peak below the floor); any
                # speech attempt prints the best score seen this chunk vs the
                # configured threshold.
                if not self._detected and chunk_peak >= 0.1:
                    logger.info(
                        "[WAKEWORD] Browser stream peak score=%.3f (threshold=%.2f, vad=%.2f) — no trigger",
                        chunk_peak,
                        self._listener.threshold,
                        self._listener.vad_threshold,
                    )

                consumed = n_complete * self._frame_size
                self._pcm_buffer = self._pcm_buffer[consumed:]

                return self._detected, self._last_label, round(self._last_score, 4)
            except ValueError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Wake-word chunk processing failed: {exc}") from exc

    def _decode_wav_to_pcm(self, wav_bytes: bytes) -> np.ndarray:
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                raw = wav_file.readframes(wav_file.getnframes())

            if sample_width != 2:
                raise ValueError(f"Unsupported WAV sample width: {sample_width * 8}-bit")
            audio = np.frombuffer(raw, dtype=np.int16)
            if channels > 1:
                audio = audio.reshape(-1, channels)[:, 0]

            if sample_rate != self.sample_rate and audio.size > 0:
                # Browser AudioContext sample rate can differ from requested 16k.
                # Resample client chunks to the detector's expected sample rate.
                src = audio.astype(np.float32)
                target_len = max(1, int(round(len(src) * self.sample_rate / sample_rate)))
                xp = np.linspace(0.0, 1.0, num=len(src), endpoint=False)
                xq = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
                resampled = np.interp(xq, xp, src)
                audio = np.clip(resampled, -32768, 32767).astype(np.int16)

            return audio
        except (wave.Error, EOFError) as exc:
            raise ValueError(f"Invalid WAV chunk for wake-word detection: {exc}") from exc
