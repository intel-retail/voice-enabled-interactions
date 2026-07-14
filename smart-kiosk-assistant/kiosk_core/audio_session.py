import logging
import tempfile
import threading
import time
import wave
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, Queue
import re
from typing import Callable
from uuid import uuid4

import numpy as np
import sounddevice as sd

from kiosk_core import config
from kiosk_core.agent_client import AgentClient
from kiosk_core.analyzer_client import AnalyzerClient
from kiosk_core.models import FileSessionStartRequest, SessionStartRequest
from kiosk_core.pipeline_latency import (
    AgentSpan, AsrSpan, LlmSpan, PipelineLatencyStore, RetrievalSpan,
    TtsSpan, TurnTrace, WallTimes, pipeline_store,
)
from kiosk_core.rag_client import RagClient
from kiosk_core.tts_client import TtsClient


logger = logging.getLogger(__name__)
_SENTENCE_PATTERN = re.compile(r"^(.+?[.!?,:;](?:[\"')\]]+)?)(?:\s+|$)", re.DOTALL)
# Whisper hallucination tokens to strip from transcripts
_WHISPER_JUNK = re.compile(
    r"\[(?:BLANK_AUDIO|Music|Noise|Applause|Laughter|Silence|Background Music|noise|music)\]",
    re.IGNORECASE,
)

# Domain vocabulary for the semantic fallback in _filter_target_speaker.
# When the primary customer is silent for an entire chunk this set is used
# to decide whether a background speaker said something kiosk-relevant enough
# to warrant re-assigning the primary (e.g. a new customer stepped up).
_DOMAIN_KEYWORDS: frozenset[str] = frozenset({
    "order", "orders", "ordering", "menu", "item", "items",
    "burger", "pizza", "sandwich", "wrap", "salad", "combo",
    "fries", "drink", "water", "coffee", "tea", "juice", "soda",
    "price", "cost", "how much", "pay", "payment", "card", "cash",
    "checkout", "bill", "receipt", "change",
    "ticket", "seat", "flight", "hotel", "book", "booking", "reserve",
    "help", "assist", "please", "want", "need", "like", "get",
})

class BaseAudioSession:
    def __init__(
        self,
        request: SessionStartRequest,
        on_complete: Callable[[str], None] | None = None,
    ):
        self.session_id = str(uuid4())
        # Persistent agent session ID — reused across all voice turns in the same
        # conversation so the ADK agent retains order state between mic presses.
        # Falls back to the audio session UUID if no conversation_id was supplied.
        self.agent_session_id: str = request.conversation_id or self.session_id
        self.request = request
        self.on_complete = on_complete
        self.client = AnalyzerClient(request.analyzer_url)
        self.rag_client = RagClient(request.rag_url)
        self.tts_client = TtsClient(request.tts_url)
        # Agent client is used when the ordering feature is enabled.
        # All turns go through the agent — it decides Q&A vs ordering.
        if config.ORDERING_ENABLED:
            agent_url = getattr(request, "agent_url", None) or config.DEFAULT_AGENT_URL
            self.agent_client: AgentClient | None = AgentClient(agent_url)
            logger.info("[SESSION] Agent routing enabled → %s", agent_url)
        else:
            self.agent_client = None
        self.created_at = datetime.now(UTC)
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.status = "created"
        self.end_reason: str | None = None
        self.error: str | None = None
        self.transcript_parts: list[str] = []
        self.response_parts: list[str] = []
        self.tts_audio_segments: list[dict[str, object]] = []
        self.tts_errors: list[str] = []
        self.stop_requested_at: datetime | None = None

        # ── Primary-speaker lock-on ────────────────────────────────────────────
        # The audio-analyzer does not set is_primary on segments. kiosk-core
        # locks onto the first speaker label seen in the session and treats all
        # subsequent segments from that label as the customer (primary).
        # Segments from any different label are unconditionally dropped — the
        # semantic fallback is only used before the primary is established.
        self._primary_speaker_id: str | None = None
        # Analyzer's own session_id — passed on every chunk so per-session
        # state (e.g. pyannote enrolled speaker embedding) persists across
        # the many chunked HTTP requests made in this kiosk session.
        # Initialised from our own session_id; the analyzer echoes/normalises
        # it via the X-Session-ID response header, which we then reuse.
        self._analyzer_session_id: str = self.session_id
        # Highest segment end-time already consumed from the analyzer. The
        # analyzer runs in append_to_session mode (session_id is reused) and
        # returns the cumulative segment list on every chunk with offset-
        # adjusted timestamps; without this cursor kiosk-core would re-append
        # every prior primary-speaker segment on each subsequent chunk,
        # duplicating utterances in transcript_parts.
        self._last_analyzer_segment_end: float = 0.0
        # ───────────────────────────────────────────────────────────────────────

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._audio_queue: Queue[np.ndarray] = Queue()
        self._thread = threading.Thread(target=self._run, name=f"mic-session-{self.session_id}", daemon=True)
        self._speech_started = False
        self._captured_samples = 0
        self._source_kind = "audio"

        # ── Pipeline timing (monotonic clock) ──────────────────────────────────
        # All _t_* fields are set during _finalize_run / _stream_rag_response.
        # Using time.monotonic() for accurate durations; datetime only for display.
        self._t_turn_start: float | None = None     # start of _finalize_run
        self._t_agent_start: float | None = None    # just before agent HTTP call
        self._t_agent_end: float | None = None      # agent reply received
        self._t_first_tts: float | None = None      # first TTS sentence queued
        self._t_last_tts: float | None = None       # last TTS segment written (in worker thread)
        self._t_turn_end: float | None = None       # after worker.join()
        self._tts_segment_count: int = 0
        # ───────────────────────────────────────────────────────────────────────

        self._frame_samples = max(1, int(self.request.sample_rate * config.DEFAULT_BLOCK_DURATION_SECONDS))
        self._frame_duration_seconds = self._frame_samples / self.request.sample_rate
        preroll_frames = max(1, int(config.DEFAULT_PREROLL_SECONDS / self._frame_duration_seconds))
        self._preroll_frames: deque[np.ndarray] = deque(maxlen=preroll_frames)
        self._session_output_dir = Path(__file__).resolve().parent.parent / "generated_audio" / self.session_id

    def start(self) -> None:
        with self._lock:
            if self.status != "created":
                raise ValueError("Session already started")
            self.status = "running"
            self.started_at = datetime.now(UTC)
        self._thread.start()

    def stop(self, reason: str = "stopped_by_api") -> None:
        with self._lock:
            if self.status not in {"running", "stopping"}:
                raise ValueError(f"Session is not running: {self.status}")
            self.status = "stopping"
            self.end_reason = reason
            self.stop_requested_at = datetime.now(UTC)
        self._stop_event.set()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            transcript = " ".join(part for part in self.transcript_parts if part).strip()
            response_text = "".join(self.response_parts).strip()
            return {
                "session_id": self.session_id,
                "source_kind": self._source_kind,
                "status": self.status,
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                "stop_requested_at": self.stop_requested_at.isoformat() if self.stop_requested_at else None,
                "end_reason": self.end_reason,
                "error": self.error,
                "speech_started": self._speech_started,
                "captured_audio_seconds": round(self._captured_samples / self.request.sample_rate, 3),
                "transcript": transcript,
                "partial_transcript": transcript,
                "transcript_parts": list(self.transcript_parts),
                "response": response_text,
                "response_parts": list(self.response_parts),
                "tts_audio_segments": [dict(segment) for segment in self.tts_audio_segments],
                "tts_errors": list(self.tts_errors),
            }

    def _run(self) -> None:
        raise NotImplementedError

    def _process_frame_stream(self, frame_iterator) -> tuple[str, str | None]:
        chunk_frames: list[np.ndarray] = []
        silence_run_seconds = 0.0
        final_status = "completed"
        end_reason = self.end_reason or "completed"

        try:
            for frame in frame_iterator:
                if self._stop_event.is_set():
                    break

                rms = self._rms(frame)
                is_speech = rms >= self.request.silence_threshold

                if not self._speech_started:
                    if is_speech:
                        self._speech_started = True
                        while self._preroll_frames:
                            buffered = self._preroll_frames.popleft()
                            chunk_frames.append(buffered)
                            self._captured_samples += len(buffered)
                        chunk_frames.append(frame)
                        self._captured_samples += len(frame)
                    else:
                        self._preroll_frames.append(frame)
                    continue

                chunk_frames.append(frame)
                self._captured_samples += len(frame)

                if is_speech:
                    silence_run_seconds = 0.0
                else:
                    silence_run_seconds += self._frame_duration_seconds

                if self._chunk_duration_seconds(chunk_frames) >= self.request.chunk_seconds:
                    self._flush_chunk(chunk_frames)
                    chunk_frames = []
                    silence_run_seconds = 0.0

                if silence_run_seconds >= self.request.silence_timeout_seconds:
                    end_reason = "silence_timeout"
                    break

                if (self._captured_samples / self.request.sample_rate) >= self.request.max_session_seconds:
                    end_reason = "max_duration_reached"
                    break

        except Exception as exc:
            final_status = "failed"
            end_reason = "error"
            with self._lock:
                self.error = str(exc)
            logger.exception("Audio session %s failed", self.session_id)

        # Final flush is intentionally outside the main try/except so that a
        # transient ASR error at the very end doesn't flip final_status to
        # "failed" and cause _finalize_run to skip RAG for an otherwise good
        # transcript.  Errors here are logged but treated as non-fatal.
        if chunk_frames and self._speech_started:
            try:
                self._flush_chunk(chunk_frames)
            except Exception as exc:
                logger.warning(
                    "Audio session %s: final flush failed (non-fatal): %s",
                    self.session_id, exc,
                )

        return final_status, end_reason

    def _finalize_run(self, final_status: str, end_reason: str) -> None:
        # Attempt RAG whenever there is a transcript, even if the session
        # ended with an error mid-stream (e.g. a transient ASR failure on one
        # chunk).  Only skip entirely when NO audio was captured at all.
        self._t_turn_start = time.monotonic()
        transcript = " ".join(part for part in self.transcript_parts if part).strip()
        if transcript:
            try:
                self._stream_rag_response(transcript)
            except Exception as exc:
                with self._lock:
                    self.error = str(exc)
                logger.exception("RAG query failed for session %s", self.session_id)
        elif final_status == "completed":
            self._synthesize_response("How can I help you?")

        with self._lock:
            if final_status == "completed" and self.end_reason == "stopped_by_api":
                end_reason = "stopped_by_api"
            self.status = final_status
            self.completed_at = datetime.now(UTC)
            self.end_reason = end_reason

        logger.info(
            "Session %s ended with reason=%s transcript=%s",
            self.session_id,
            self.end_reason,
            " ".join(self.transcript_parts).strip(),
        )
        if self.on_complete is not None:
            self.on_complete(self.session_id)

    def _synthesize_response(self, text: str) -> None:
        """Speak a fixed response directly via TTS, without calling RAG."""
        with self._lock:
            self.response_parts.append(text)
        sentence_queue: Queue[tuple[int | None, str | None]] = Queue()
        worker = threading.Thread(target=self._tts_worker, args=(sentence_queue,), daemon=True)
        worker.start()
        sentence_queue.put((1, text))
        sentence_queue.put((None, None))
        worker.join()

    def _stream_rag_response(self, transcript: str) -> None:
        pending_text = ""
        sentence_queue: Queue[tuple[int | None, str | None]] = Queue()
        worker = threading.Thread(target=self._tts_worker, args=(sentence_queue,), daemon=True)
        worker.start()

        history = list(getattr(self.request, "history", []) or [])

        # Route through the ordering agent when enabled; fall back to direct RAG.
        if self.agent_client is not None:
            logger.info("[SESSION] Routing turn to agent: session=%s (conv=%s) message=%r",
                        self.session_id, self.agent_session_id, transcript[:80])
        if self.agent_client is not None:
            logger.info("[SESSION] Routing turn to agent: session=%s (conv=%s) message=%r",
                        self.session_id, self.agent_session_id, transcript[:80])
            token_source = self.agent_client.get_reply(
                transcription=transcript,
                session_id=self.agent_session_id,  # persistent across voice turns
                user_id=getattr(self.request, "user_id", None) or "kiosk-user",
                history=history,
            )
            label = "Agent"
        else:
            token_source = self.rag_client.stream_answer(transcript, history=history)
            label = "RAG"

        print(f"\n{label} response for session {self.session_id}:\n", end="", flush=True)
        sentence_index = 0
        _first_token_seen = False
        _tool_calls: list[str] = []
        # t_agent_start set here — generator body (HTTP call) runs on first iteration
        if self.agent_client is not None:
            self._t_agent_start = time.monotonic()
        try:
            for token in token_source:
                # Handle metadata sentinel from AgentClient BEFORE appending to response_parts
                # to avoid dict items in response_parts (which causes TypeError in snapshot())
                if isinstance(token, dict) and "_tool_calls" in token:
                    _tool_calls = token["_tool_calls"]
                    continue

                with self._lock:
                    self.response_parts.append(token)
                print(token, end="", flush=True)

                if not _first_token_seen:
                    _first_token_seen = True
                    self._t_agent_end = time.monotonic()

                pending_text += token
                complete_sentences, pending_text = self._drain_complete_sentences(pending_text)
                for sentence in complete_sentences:
                    sentence_index += 1
                    if sentence_index == 1:
                        self._t_first_tts = time.monotonic()
                    sentence_queue.put((sentence_index, sentence))

            trailing_text = pending_text.strip()
            if trailing_text:
                sentence_index += 1
                if sentence_index == 1:
                    self._t_first_tts = time.monotonic()
                sentence_queue.put((sentence_index, trailing_text))

            # If agent_end wasn't set (empty reply), set it now
            if self._t_agent_start is not None and self._t_agent_end is None:
                self._t_agent_end = time.monotonic()

        finally:
            sentence_queue.put((None, None))
            worker.join()
            self._t_turn_end = time.monotonic()
            self._tts_segment_count = sentence_index
            print(flush=True)

        # ── Record pipeline turn trace ──────────────────────────────────────
        self._record_turn_trace(_tool_calls)

    def _record_turn_trace(self, tool_calls: list[str]) -> None:
        """Build and persist a TurnTrace for the completed voice turn."""
        t0 = self._t_turn_start
        t_agent_s = self._t_agent_start
        t_agent_e = self._t_agent_end
        t_first = self._t_first_tts
        t_last = self._t_last_tts
        t_end = self._t_turn_end

        def _ms(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return None
            return round((b - a) * 1000, 1)

        # ASR = from turn_start to when agent was called (includes VAD→transcribe)
        asr_ms = _ms(t0, t_agent_s)
        # Agent TTFT = from agent_start to when first token (reply) arrived
        ttft_ms = _ms(t_agent_s, t_agent_e)
        # Agent total = from agent_start to last TTS segment done (whole orchestration)
        agent_total_ms = _ms(t_agent_s, t_end)
        # TTS = from first sentence queued to last segment written
        tts_ms = _ms(t_first, t_last)
        # Time to first audio = from agent call start to first TTS sentence queued
        ttfa_ms = _ms(t_agent_s, t_first)
        # Wall E2E: measured end-to-end (turn_start → turn_end), never summed
        wall_total_ms = _ms(t0, t_end)

        retrieval_invoked = any(
            "retrieval" in tc.lower() or "knowledge" in tc.lower() or "lookup" in tc.lower()
            for tc in tool_calls
        )

        trace = TurnTrace(
            turn_id=self.session_id,
            conversation_id=self.agent_session_id,
            started_at=self.started_at.isoformat() if self.started_at else datetime.now(UTC).isoformat(),
            ended_at=datetime.now(UTC).isoformat(),
            wall=WallTimes(
                turn_total_ms=wall_total_ms,
                time_to_first_audio_ms=ttfa_ms,
            ),
            asr=AsrSpan(ms=asr_ms),
            agent=AgentSpan(
                ttft_ms=ttft_ms,
                total_ms=agent_total_ms,
                retrieval=RetrievalSpan(invoked=retrieval_invoked),
                llm=LlmSpan(device="GPU"),
            ),
            tts=TtsSpan(
                ms=tts_ms,
                segments=self._tts_segment_count,
                overlapped_with_agent=True,
            ),
        )
        pipeline_store.record(trace)
        logger.info(
            "[PIPELINE] turn=%s wall=%.0fms asr=%.0fms ttft=%.0fms tts=%.0fms retrieval=%s tools=%s",
            self.session_id,
            wall_total_ms or 0,
            asr_ms or 0,
            ttft_ms or 0,
            tts_ms or 0,
            retrieval_invoked,
            tool_calls,
        )

    @staticmethod
    def _drain_complete_sentences(buffer: str) -> tuple[list[str], str]:
        sentences: list[str] = []
        remaining = buffer
        while True:
            match = _SENTENCE_PATTERN.match(remaining.lstrip())
            if match is None:
                break
            sentence = match.group(1).strip()
            if sentence:
                sentences.append(sentence)
            remaining = remaining.lstrip()[match.end() :]
        return sentences, remaining

    def _tts_worker(self, sentence_queue: Queue[tuple[int | None, str | None]]) -> None:
        while True:
            sentence_index, sentence = sentence_queue.get()
            if sentence_index is None or sentence is None:
                return

            output_path = self._session_output_dir / f"response_{sentence_index:03d}.wav"
            try:
                self.tts_client.synthesize_to_file(
                    text=sentence,
                    output_path=str(output_path),
                    model=self.request.tts_model,
                    voice=self.request.tts_voice,
                    language=self.request.tts_language,
                    instructions=self.request.tts_instructions,
                )
                with self._lock:
                    self._t_last_tts = time.monotonic()
                    self.tts_audio_segments.append(
                        {
                            "index": sentence_index,
                            "text": sentence,
                            "audio_file": str(output_path),
                        }
                    )
            except Exception as exc:
                logger.exception("TTS synthesis failed for session %s sentence %s", self.session_id, sentence_index)
                with self._lock:
                    self.tts_errors.append(f"sentence {sentence_index}: {exc}")

    def _on_audio(self, indata, frames, time, status) -> None:
        del frames, time
        if status:
            logger.warning("Audio callback status for %s: %s", self.session_id, status)
        self._audio_queue.put(indata[:, 0].copy())

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        samples = frame.astype(np.float32)
        return float(np.sqrt(np.mean(samples * samples)))

    def _chunk_duration_seconds(self, frames: list[np.ndarray]) -> float:
        total_samples = sum(len(frame) for frame in frames)
        return total_samples / self.request.sample_rate

    def _flush_chunk(self, frames: list[np.ndarray]) -> None:
        audio = np.concatenate(frames, axis=0)
        temp_path = self._write_temp_wav(audio)
        try:
            duration = len(audio) / self.request.sample_rate
            logger.info(
                "[CHUNK] session=%s | flushing %.2fs of audio, diarization=%s",
                self.session_id, duration, config.DEFAULT_DIARIZATION_ENABLED,
            )
            payload = self.client.transcribe_file(
                temp_path,
                language=self.request.language,
                temperature=self.request.temperature,
                diarization=config.DEFAULT_DIARIZATION_ENABLED,
                session_id=self._analyzer_session_id,
            )
            # Latch the analyzer's assigned session id from the first
            # response so subsequent chunks reuse the same server-side
            # session (and its enrolled primary voice embedding).
            assigned = (
                payload.get("_analyzer_session_id")
                if isinstance(payload, dict)
                else None
            )
            if assigned and assigned != self._analyzer_session_id:
                logger.info(
                    "[CHUNK] session=%s | analyzer session pinned to %s",
                    self.session_id, assigned,
                )
                self._analyzer_session_id = assigned
            segments: list[dict] = payload.get("segments", []) if isinstance(payload, dict) else []
            raw_text = str(payload.get("text", "")).strip() if isinstance(payload, dict) else str(payload).strip()

            logger.info(
                "[CHUNK] session=%s | audio-analyzer response: %d segment(s), flat_text=%r",
                self.session_id, len(segments), raw_text[:120],
            )

            # ── Dedupe against analyzer's cumulative session state ──────────
            # The analyzer reuses our session_id in append_to_session mode and
            # returns EVERY segment ever produced for the session (with
            # timestamps offset by the accumulated duration). Keep only the
            # segments that start after the last end-time we've already seen;
            # otherwise every prior utterance gets re-appended to the
            # transcript on each new chunk.
            if segments:
                fresh_segments = [
                    s for s in segments
                    if float(s.get("end", 0.0)) > self._last_analyzer_segment_end + 1e-3
                ]
                if len(fresh_segments) != len(segments):
                    logger.info(
                        "[CHUNK] session=%s | deduped %d cumulative segment(s) → %d fresh (cursor=%.2fs)",
                        self.session_id,
                        len(segments) - len(fresh_segments),
                        len(fresh_segments),
                        self._last_analyzer_segment_end,
                    )
                segments = fresh_segments
                if segments:
                    self._last_analyzer_segment_end = max(
                        self._last_analyzer_segment_end,
                        max(float(s.get("end", 0.0)) for s in segments),
                    )
                    # Rebuild raw_text from fresh segments so the flat-text
                    # fallback (used when diarization is off or returns no
                    # segments) is also free of the cumulative duplicates.
                    raw_text = " ".join(
                        s.get("text", "").strip() for s in segments if s.get("text", "").strip()
                    ).strip()
                else:
                    raw_text = ""

            if segments and config.DEFAULT_DIARIZATION_ENABLED:
                text = self._filter_target_speaker(segments)
            else:
                if config.DEFAULT_DIARIZATION_ENABLED and not segments:
                    logger.info(
                        "[CHUNK] session=%s | diarization enabled but no segments returned — using flat text",
                        self.session_id,
                    )
                text = raw_text

            if text:
                # Strip Whisper hallucination tokens (e.g. [BLANK_AUDIO], [Music])
                text = _WHISPER_JUNK.sub("", text).strip()
            if text:
                logger.info(
                    "[CHUNK] session=%s | appending to transcript: %r",
                    self.session_id, text[:120],
                )
                with self._lock:
                    self.transcript_parts.append(text)
            else:
                logger.info(
                    "[CHUNK] session=%s | chunk produced no usable text (filtered or empty)",
                    self.session_id,
                )
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def _filter_target_speaker(self, segments: list[dict]) -> str:
        """Filter diarized segments to keep only the primary customer's speech.

        Primary speaker is determined in order of precedence:
        1. Analyzer-provided ``is_primary`` flag on the segment — honoured
           directly when present, so the analyzer's own enrollment logic wins.
        2. First-speaker lock-on — the first speaker label seen in this session
           is treated as the customer.  Every subsequent segment from that label
           is kept; any other label is unconditionally dropped.

        Semantic fallback is used only before the primary speaker is
        established (i.e. on the very first chunk that has no clear speaker
        label), so a legitimate opening utterance is never lost.
        """
        if not segments:
            return ""

        # Defensive read — guard against subclasses or test stubs that may not
        # have called BaseAudioSession.__init__.
        primary_speaker_id: str | None = getattr(self, "_primary_speaker_id", None)

        # ── Honor analyzer-provided is_primary when present ──────────────────
        if any("is_primary" in s for s in segments):
            primary_segments = [s for s in segments if s.get("is_primary")]
            if primary_segments:
                # Also update / initialise the lock-on label from the first
                # primary segment so label-based filtering stays in sync.
                label = primary_segments[0].get("speaker", "")
                if label and primary_speaker_id is None:
                    self._primary_speaker_id = label
                    logger.info(
                        "[SPEAKER-LOCK] session=%s | primary speaker locked → %s (is_primary flag)",
                        self.session_id, label,
                    )
                final_text = " ".join(seg.get("text", "") for seg in primary_segments).strip()
                logger.info(
                    "[SPEAKER-FILTER] session=%s | is_primary path: kept=%d | final_text=%r",
                    self.session_id, len(primary_segments), final_text[:120],
                )
                return final_text

        # ── Lock on to the first speaker seen in this session ────────────────
        if primary_speaker_id is None:
            for seg in segments:
                label = seg.get("speaker", "")
                if label:
                    self._primary_speaker_id = label
                    primary_speaker_id = label
                    logger.info(
                        "[SPEAKER-LOCK] session=%s | primary speaker locked → %s (first speaker rule)",
                        self.session_id, primary_speaker_id,
                    )
                    break

        # ── Classify segments as primary / non-primary ───────────────────────
        if primary_speaker_id:
            kept_segments = [s for s in segments if s.get("speaker") == primary_speaker_id]
            discarded_segments = [s for s in segments if s.get("speaker") != primary_speaker_id]
        else:
            # No speaker label at all — treat everything as potentially primary
            kept_segments = []
            discarded_segments = list(segments)

        logger.info(
            "[SPEAKER-FILTER] session=%s | processing %d segment(s): %d primary, %d non-primary",
            self.session_id, len(segments), len(kept_segments), len(discarded_segments),
        )
        for i, segment in enumerate(segments):
            is_primary = primary_speaker_id is not None and segment.get("speaker") == primary_speaker_id
            logger.info(
                "[SPEAKER-FILTER] session=%s | seg[%d] speaker=%s is_primary=%s → %s | text=%r",
                self.session_id, i, segment.get("speaker", "UNKNOWN"), is_primary,
                "KEEP" if is_primary else "DISCARD", segment.get("text", "")[:80],
            )

        # ── Semantic fallback — only before primary is established ───────────
        if not kept_segments and discarded_segments:
            if primary_speaker_id:
                # Primary is known — non-primary utterances are noise, drop them.
                logger.info(
                    "[SPEAKER-FILTER] session=%s | non-primary speech only, primary=%s — chunk DROPPED (no fallback)",
                    self.session_id, primary_speaker_id,
                )
            else:
                # Primary not yet established — run semantic fallback so the
                # very first domain utterance (before a clean speaker label
                # arrives) is not silently discarded.
                logger.info(
                    "[SPEAKER-FILTER] session=%s | primary not yet established — running semantic fallback on %d segment(s)",
                    self.session_id, len(discarded_segments),
                )
                best_score = 0.0
                best_segment: dict | None = None
                for segment in discarded_segments:
                    words = segment.get("text", "").lower().split()
                    if not words:
                        continue
                    overlap = sum(1 for w in words if w in _DOMAIN_KEYWORDS)
                    score = overlap / max(len(words), 1)
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | fallback score speaker=%s score=%.2f | text=%r",
                        self.session_id, segment.get("speaker", "UNKNOWN"), score, segment.get("text", "")[:80],
                    )
                    if score > best_score:
                        best_score = score
                        best_segment = segment

                if best_segment is not None and best_score >= config.DEFAULT_SEMANTIC_FALLBACK_THRESHOLD:
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | semantic fallback ACCEPTED speaker=%s score=%.2f | text=%r",
                        self.session_id, best_segment.get("speaker", "UNKNOWN"), best_score, best_segment.get("text", "")[:80],
                    )
                    kept_segments = [best_segment]
                else:
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | semantic fallback found no domain match (best_score=%.2f, threshold=%.2f) → chunk DROPPED",
                        self.session_id, best_score, config.DEFAULT_SEMANTIC_FALLBACK_THRESHOLD,
                    )

        final_text = " ".join(seg.get("text", "") for seg in kept_segments).strip()
        logger.info(
            "[SPEAKER-FILTER] session=%s | RESULT: kept=%d dropped=%d | final_text=%r",
            self.session_id,
            len(kept_segments),
            len(discarded_segments) - (1 if kept_segments and discarded_segments else 0),
            final_text[:120],
        )
        return final_text

    def _write_temp_wav(self, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(prefix=f"{self.session_id}-", suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name

        with wave.open(temp_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.request.sample_rate)
            wav_file.writeframes(audio.astype(np.int16).tobytes())

        return temp_path


class BrowserStreamSession(BaseAudioSession):
    """Session that receives audio chunks pushed from the browser via HTTP.

    Call push_audio(wav_bytes) from the HTTP handler each time a chunk arrives.
    The session applies the same RMS silence detection and chunk-flushing logic
    as MicrophoneSession.  It ends automatically when:
      - silence_timeout_seconds of silence follows detected speech, OR
      - max_session_seconds of captured audio have been processed, OR
      - stop() is called explicitly (e.g. user clicks stop-recording in browser).
    """

    def __init__(
        self,
        request: SessionStartRequest,
        on_complete: Callable[[str], None] | None = None,
    ):
        super().__init__(request=request, on_complete=on_complete)
        self._thread = threading.Thread(target=self._run, name=f"browser-session-{self.session_id}", daemon=True)
        self._source_kind = "browser"
        # Sentinel: None means end-of-stream
        self._push_queue: Queue[np.ndarray | None] = Queue()

    def push_audio(self, wav_bytes: bytes) -> None:
        """Called from the HTTP handler for each incoming audio chunk."""
        audio = np.frombuffer(wav_bytes, dtype=np.int16)
        # Split into frame-sized pieces so _process_frame_stream sees uniform frames
        for start in range(0, len(audio), self._frame_samples):
            frame = audio[start : start + self._frame_samples]
            if len(frame) > 0:
                self._push_queue.put(frame.copy())

    def signal_end(self) -> None:
        """Signal that the browser has stopped recording (enqueue sentinel)."""
        self._push_queue.put(None)

    def _run(self) -> None:
        final_status = "completed"
        end_reason = self.end_reason or "completed"
        try:
            final_status, end_reason = self._process_frame_stream(self._iter_push_frames())
            if final_status == "completed" and not self._speech_started:
                end_reason = "no_speech_detected"
        except Exception as exc:
            final_status = "failed"
            end_reason = "error"
            with self._lock:
                self.error = str(exc)
            logger.exception("Browser stream session %s failed", self.session_id)
        finally:
            self._finalize_run(final_status, end_reason)

    def _iter_push_frames(self):
        while not self._stop_event.is_set():
            try:
                frame = self._push_queue.get(timeout=0.25)
            except Empty:
                continue
            if frame is None:
                # End-of-stream sentinel from signal_end()
                break
            yield frame


class MicrophoneSession(BaseAudioSession):
    def __init__(
        self,
        request: SessionStartRequest,
        on_complete: Callable[[str], None] | None = None,
    ):
        super().__init__(request=request, on_complete=on_complete)
        self._thread = threading.Thread(target=self._run, name=f"mic-session-{self.session_id}", daemon=True)
        self._source_kind = "microphone"

    def _run(self) -> None:
        final_status = "completed"
        end_reason = self.end_reason or "completed"
        try:
            with sd.InputStream(
                samplerate=self.request.sample_rate,
                blocksize=self._frame_samples,
                channels=1,
                dtype="int16",
                device=self.request.device,
                callback=self._on_audio,
            ):
                def iter_frames():
                    while not self._stop_event.is_set():
                        try:
                            yield self._audio_queue.get(timeout=0.25)
                        except Empty:
                            continue

                final_status, end_reason = self._process_frame_stream(iter_frames())
        except Exception as exc:
            final_status = "failed"
            end_reason = "error"
            with self._lock:
                self.error = str(exc)
            logger.exception("Microphone session %s failed", self.session_id)
        finally:
            self._finalize_run(final_status, end_reason)


class FileAudioSession(BaseAudioSession):
    def __init__(
        self,
        request: FileSessionStartRequest,
        audio_file_path: str,
        on_complete: Callable[[str], None] | None = None,
    ):
        super().__init__(request=request, on_complete=on_complete)
        self.request = request
        self.audio_file_path = audio_file_path
        self._thread = threading.Thread(target=self._run, name=f"file-session-{self.session_id}", daemon=True)
        self._source_kind = "file"

    def _run(self) -> None:
        final_status = "completed"
        end_reason = self.end_reason or "completed"
        try:
            final_status, end_reason = self._process_frame_stream(self._iter_file_frames())
            if final_status == "completed" and not self._speech_started:
                end_reason = "no_speech_detected"
        except Exception as exc:
            final_status = "failed"
            end_reason = "error"
            with self._lock:
                self.error = str(exc)
            logger.exception("File session %s failed", self.session_id)
        finally:
            Path(self.audio_file_path).unlink(missing_ok=True)
            self._finalize_run(final_status, end_reason)

    def _iter_file_frames(self):
        with wave.open(self.audio_file_path, "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()

            if sample_width != 2:
                raise ValueError("Only 16-bit PCM WAV files are supported for file-based testing")
            if sample_rate != self.request.sample_rate:
                raise ValueError(
                    f"Uploaded WAV sample rate {sample_rate} does not match requested sample_rate {self.request.sample_rate}"
                )

            while not self._stop_event.is_set():
                raw = wav_file.readframes(self._frame_samples)
                if not raw:
                    break

                frame = np.frombuffer(raw, dtype=np.int16)
                if channels > 1:
                    frame = frame.reshape(-1, channels)[:, 0]

                if len(frame) == 0:
                    continue

                yield frame.copy()

                if self.request.realtime_factor > 0:
                    time.sleep((len(frame) / self.request.sample_rate) / self.request.realtime_factor)
