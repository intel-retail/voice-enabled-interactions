import logging
import tempfile
import threading
import time
import unicodedata
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

        # Speaker diarization state — persists across chunk boundaries for
        # the lifetime of this session.  Reset when the session is recreated.
        self.primary_speaker_id: str | None = None
        self.pending_segments: list[dict] = []

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
                "primary_speaker_id": self.primary_speaker_id,
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
        # If speaker-filter buffered segments but primary was never locked
        # (pyannote returned UNKNOWN for the whole session — common for short
        # utterances), flush pending as primary speech so the transcript is not lost.
        if self.pending_segments and not self.primary_speaker_id:
            pending_text = " ".join(
                seg.get("text", "") for seg in self.pending_segments
            ).strip()
            if pending_text:
                pending_text = _WHISPER_JUNK.sub("", pending_text).strip()
            if pending_text:
                logger.info(
                    "[SPEAKER-FILTER] session=%s | session end — flushing %d pending UNKNOWN segment(s) as primary transcript: %r",
                    self.session_id, len(self.pending_segments), pending_text[:120],
                )
                with self._lock:
                    self.transcript_parts.append(pending_text)
            self.pending_segments = []

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
            )
            segments: list[dict] = payload.get("segments", []) if isinstance(payload, dict) else []
            raw_text = str(payload.get("text", "")).strip() if isinstance(payload, dict) else str(payload).strip()

            logger.info(
                "[CHUNK] session=%s | audio-analyzer response: %d segment(s), flat_text=%r",
                self.session_id, len(segments), raw_text[:120],
            )

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

    @staticmethod
    def _meaningful_char_count(text: str) -> int:
        """Count characters that are not whitespace and not Unicode punctuation."""
        return sum(
            1 for ch in text
            if not ch.isspace() and not unicodedata.category(ch).startswith("P")
        )

    def _filter_target_speaker(self, segments: list[dict]) -> str:
        """Filter diarized segments to keep only the primary customer's speech.

        Phase 1 — lock-on + pending-buffer:
          - UNKNOWN speaker before primary is locked → buffer in pending_segments.
          - First meaningful KNOWN speaker → lock as primary; flush pending.
          - UNKNOWN after lock → treat as primary continuation (inherit).
          - KNOWN != primary → discard.

        Phase 2 — semantic fallback:
          If *no* segment was kept (primary was silent the whole chunk), score
          each discarded segment against DOMAIN_KEYWORDS.  The best-scoring
          segment above the threshold wins and reassigns primary_speaker_id.
        """
        if not segments:
            return ""

        kept_segments: list[dict] = []
        discarded_segments: list[dict] = []

        logger.info(
            "[SPEAKER-FILTER] session=%s | processing %d segment(s), primary_speaker=%s",
            self.session_id, len(segments),
            self.primary_speaker_id if self.primary_speaker_id else "NOT_SET",
        )

        for i, segment in enumerate(segments):
            speaker: str | None = segment.get("speaker")
            text: str = segment.get("text", "")

            # ── UNKNOWN speaker (Pyannote gap / silence) ─────────────────────
            if not speaker:
                if self.primary_speaker_id is not None:
                    # Post-lock: inherit primary, give benefit of the doubt.
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | seg[%d] UNKNOWN → inheriting primary=%s → KEEP | text=%r",
                        self.session_id, i, self.primary_speaker_id, text[:80],
                    )
                    kept_segments.append(segment)
                elif len(self.pending_segments) >= 2:
                    # Pyannote never assigned a speaker (short audio / single speaker).
                    # After 2 buffered chunks, accept UNKNOWN as the primary.
                    self.primary_speaker_id = "SPEAKER_00"
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | seg[%d] UNKNOWN — pending limit reached, "
                        "locking SPEAKER_00 as primary, flushing %d pending segment(s) | text=%r",
                        self.session_id, i, len(self.pending_segments), text[:80],
                    )
                    kept_segments.extend(self.pending_segments)
                    self.pending_segments = []
                    kept_segments.append(segment)
                else:
                    # Pre-lock: buffer for retroactive assignment.
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | seg[%d] UNKNOWN, no primary yet → BUFFERED (pending=%d) | text=%r",
                        self.session_id, i, len(self.pending_segments) + 1, text[:80],
                    )
                    self.pending_segments.append(segment)
                continue

            # ── KNOWN speaker ─────────────────────────────────────────────────
            if self.primary_speaker_id is None:
                if self._meaningful_char_count(text) > 1:
                    # Lock primary speaker on first meaningful utterance.
                    self.primary_speaker_id = speaker
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | seg[%d] speaker=%s → PRIMARY LOCKED ✓ | flushing %d pending segment(s) | text=%r",
                        self.session_id, i, speaker, len(self.pending_segments), text[:80],
                    )
                    # Retroactively flush all buffered unknown segments.
                    if self.pending_segments:
                        logger.info(
                            "[SPEAKER-FILTER] session=%s | flushing %d pending segment(s) → assigned to primary=%s",
                            self.session_id, len(self.pending_segments), speaker,
                        )
                    kept_segments.extend(self.pending_segments)
                    self.pending_segments = []
                    kept_segments.append(segment)
                else:
                    # Noise token or single punctuation — discard.
                    logger.info(
                        "[SPEAKER-FILTER] session=%s | seg[%d] speaker=%s → noise/punctuation only, no lock → DISCARD | text=%r",
                        self.session_id, i, speaker, text[:80],
                    )
                    discarded_segments.append(segment)
            elif speaker == self.primary_speaker_id:
                logger.info(
                    "[SPEAKER-FILTER] session=%s | seg[%d] speaker=%s == primary → KEEP | text=%r",
                    self.session_id, i, speaker, text[:80],
                )
                kept_segments.append(segment)
            else:
                logger.info(
                    "[SPEAKER-FILTER] session=%s | seg[%d] speaker=%s != primary (%s) → DISCARD (secondary voice) | text=%r",
                    self.session_id, i, speaker, self.primary_speaker_id, text[:80],
                )
                discarded_segments.append(segment)

        # ── Phase 2: Semantic fallback ────────────────────────────────────────
        if not kept_segments and discarded_segments:
            logger.info(
                "[SPEAKER-FILTER] session=%s | primary was silent this chunk — running semantic fallback on %d discarded segment(s)",
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
                new_speaker = best_segment.get("speaker")
                logger.info(
                    "[SPEAKER-FILTER] session=%s | semantic fallback ACCEPTED speaker=%s score=%.2f — PRIMARY REASSIGNED ✓ | text=%r",
                    self.session_id, new_speaker, best_score, best_segment.get("text", "")[:80],
                )
                self.primary_speaker_id = new_speaker
                kept_segments = [best_segment]
            else:
                logger.info(
                    "[SPEAKER-FILTER] session=%s | semantic fallback found no domain match (best_score=%.2f, threshold=%.2f) → chunk DROPPED",
                    self.session_id, best_score, config.DEFAULT_SEMANTIC_FALLBACK_THRESHOLD,
                )

        final_text = " ".join(seg.get("text", "") for seg in kept_segments).strip()
        logger.info(
            "[SPEAKER-FILTER] session=%s | RESULT: kept=%d dropped=%d primary=%s | final_text=%r",
            self.session_id,
            len(kept_segments),
            len(discarded_segments) - (1 if kept_segments and discarded_segments else 0),
            self.primary_speaker_id,
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
