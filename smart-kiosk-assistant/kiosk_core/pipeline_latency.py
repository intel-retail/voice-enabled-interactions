"""
Turn-level AI pipeline latency tracker.

Captures one TurnTrace per completed voice turn and holds the last N in a
thread-safe ring buffer.  Exposed via GET /api/v1/pipeline/latest and
GET /api/v1/pipeline/recent on kiosk-core.

Design principles
─────────────────
* Wall-clock E2E is MEASURED (ended_at − started_at), never summed from stages,
  so the TTS-overlap with LLM generation is handled correctly.
* Spans are NESTED: retrieval and llm live under agent (matching runtime reality).
* An ``invoked`` flag on retrieval prevents stale latency leaking across turns.
* All durations use monotonic clock (time.monotonic); ISO timestamps use datetime.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class RetrievalSpan:
    invoked: bool = False
    ms: float | None = None


@dataclass
class McpSpan:
    ms: float | None = None        # cumulative MCP tool round-trip time (network + kiosk-core handling)
    calls: int = 0


@dataclass
class GuardSpan:
    ms: float | None = None        # cumulative truthfulness-guard processing time
    calls: int = 0


@dataclass
class TemplateSpan:
    ms: float | None = None        # deterministic reply-template render time
    calls: int = 0


@dataclass
class LlmSpan:
    ms: float | None = None        # total cumulative LLM time this turn
    ttft_ms: float | None = None   # cumulative prefill; ms - ttft_ms = decode
    calls: int = 0
    device: str = "GPU"


@dataclass
class AgentSpan:
    ttft_ms: float | None = None   # time-to-first-token (perceived latency)
    total_ms: float | None = None  # agent_start to last TTS segment done —
                                    # includes TTS-synthesis drain time, NOT
                                    # pure agent/LLM/tool time. Kept as-is
                                    # (existing consumers rely on this whole-
                                    # orchestration semantics); use stream_ms
                                    # below for the isolated figure.
    stream_ms: float | None = None  # agent_start to the end of the token
                                     # stream (before _stop_tts_workers runs)
                                     # — the actual LLM+tool round-trip a
                                     # customer's reply waits on, with TTS
                                     # drain time excluded.
    retrieval: RetrievalSpan = field(default_factory=RetrievalSpan)
    llm: LlmSpan = field(default_factory=LlmSpan)
    mcp: McpSpan = field(default_factory=McpSpan)
    guard: GuardSpan = field(default_factory=GuardSpan)
    template: TemplateSpan = field(default_factory=TemplateSpan)


@dataclass
class AsrSpan:
    ms: float | None = None
    device: str = "CPU"
    chunks: int = 0                # number of transcribe calls summed into ms
    final_flush_skipped: bool = False  # see config.DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED
    # Transcription latency: the customer's actual last word (same anchor as
    # wall.voice_to_voice_ms) to the moment a transcript covering it landed
    # from the analyzer. This is the genuine ASR compute latency on the
    # critical path -- it deliberately EXCLUDES the trailing-silence wait the
    # endpoint sits through before acting on that transcript
    # (wall.endpointing_delay_ms already reports that), so it is comparable
    # to a bare "utterance -> transcript" figure rather than inflated by a
    # design choice unrelated to ASR speed.
    #
    # Continuous-streaming mode only (KIOSK_CORE_ANALYZER_STREAMING_ENABLED).
    # None when streaming mode never ran this turn, or no post-last-word
    # transcript update was observed (the analyzer already had everything
    # before the customer finished speaking).
    transcription_latency_ms: float | None = None


@dataclass
class TtsSpan:
    ms: float | None = None
    device: str = "CPU"
    segments: int = 0
    overlapped_with_agent: bool = True   # always true — TTS runs concurrently
    # TTS time to first byte: first sentence handed to the synthesiser ->
    # that sentence's audio on disk. The stage figure that pairs with
    # agent.llm.ttft_ms; ``ms`` above is the whole synthesis drain, which
    # keeps running long after the customer has started hearing the reply.
    ttfb_ms: float | None = None


@dataclass
class WallTimes:
    """Per-turn latency, using the agreed Smart Kiosk latency vocabulary.

    The three headline spans are defined so that they always reconcile on a
    single wall clock::

        voice_to_voice_ms = endpointing_delay_ms + processing_latency_ms

    ``time_to_first_audio_ms`` was retired: it meant different spans in
    different places (sometimes from the endpoint decision, sometimes from
    the last word) and it stopped the clock when a WAV was written rather
    than at the speaker.
    """

    turn_total_ms: float | None = None
    # ── Headline: customer's last word -> first sound at the speaker ──────
    # The number a customer actually feels and the only one comparable to
    # external voice-kiosk figures. Includes the opener when the opener is
    # enabled -- check ``first_audio_was_opener`` before quoting it as a
    # pipeline figure, and prefer ``voice_to_voice_answer_ms`` if it is True.
    voice_to_voice_ms: float | None = None
    # Customer's last word -> the turn-end decision (the endpoint committing
    # on trailing silence, or the mic-release signal being processed).
    # Wall-clock, measured between two backend monotonic stamps. This is a
    # design choice (how long we deliberately wait to be sure the customer
    # finished), not compute cost, but the customer sits through it, so it is
    # part of voice_to_voice_ms.
    endpointing_delay_ms: float | None = None
    # Turn-end decision -> first sound at the speaker. This is the part that
    # is actually our pipeline: final ASR flush, agent/LLM, tools, guards and
    # TTS. The number to optimise and to compare across devices.
    processing_latency_ms: float | None = None
    # ── Same spans, but stopping on the first sound that carries the ANSWER.
    # The opener ("One moment.") is real audio and legitimately breaks the
    # silence, but it tells the customer nothing and is a cached file copy,
    # so it is reported separately rather than allowed to flatter the
    # headline. With the opener disabled these equal the two fields above.
    voice_to_voice_answer_ms: float | None = None
    processing_latency_answer_ms: float | None = None
    # True when the first sound of this turn was the canned opener. Makes a
    # near-zero processing_latency_ms self-explanatory in the trace instead
    # of looking like a pipeline result.
    first_audio_was_opener: bool = False
    # True when the opener was enabled for this turn but could not be
    # rendered, so the customer heard nothing until the answer itself. A
    # failed synthesis is remembered for a cooldown rather than for the life
    # of the process, so this can appear on some turns of a run and not
    # others; without it, an opener that had silently switched itself off was
    # indistinguishable from one that was never configured.
    opener_failed: bool = False
    # ── Sub-components of the two spans above (diagnostics) ──────────────
    # The endpoint's trailing-silence run at the instant it committed, in
    # AUDIO-domain seconds (counted from samples, not the wall clock). Kept
    # as a diagnostic for "did the shortcut fire?" -- never mix it into a
    # wall-clock arithmetic, which is what the old endpoint_wait_ms did.
    endpoint_silence_run_ms: float | None = None
    # The final chunk's real ASR round-trip: the mandatory drain-and-join of
    # the flush queue that _finalize_run blocks on right after the turn-end
    # decision. A component of processing_latency_ms, broken out because it
    # is the single largest non-obvious contributor to it.
    final_flush_wait_ms: float | None = None
    # Browser mic-release turns only: the gap between the customer's actual
    # last speech frame and the moment the flush/turn-start sequence began --
    # button-release reaction time plus any trailing buffered frames. A
    # component of endpointing_delay_ms on that path.
    post_speech_gap_ms: float | None = None
    # File-replay only (FileAudioSession): time from the first byte of the
    # fixture being fed into the pipeline to first audio out. The fields
    # above anchor on OUR OWN detector's view of the last word, which is a
    # live-mic necessity -- there is no ground truth on a live mic. A
    # fixture, unlike a live mic, HAS an independent ground truth: the exact
    # sample where the recording's real speech ends, discoverable once via a
    # silence detector run directly on the file (outside this pipeline, so it
    # is not circular). Benchmarks combine this field with that offset to get
    # a voice-to-voice number with zero dependency on this system's own VAD
    # timing. None for microphone/browser-stream sessions.
    playback_to_first_audio_ms: float | None = None
    playback_to_answer_audio_ms: float | None = None
    # Same playback anchor, stopping at the turn-end DECISION rather than at
    # audio out. This is the field that lets a benchmark split a clip-anchored
    # voice_to_voice_ms into its endpointing and processing halves without
    # ever consulting our own VAD for the start instant:
    #
    #   endpointing_delay  = playback_to_endpoint_decision_ms - true_eos_ms
    #   processing_latency = playback_to_first_audio_ms
    #                        - playback_to_endpoint_decision_ms
    #
    # where true_eos_ms comes from a silence detector run on the file itself.
    # Both halves stay on the same playback clock, so they subtract cleanly.
    playback_to_endpoint_decision_ms: float | None = None
    # True: this turn committed via the sentence-completeness shortcut
    # (KIOSK_CORE_ENDPOINT_SHORT_SECONDS). False: it fell through to the full
    # silence_timeout_seconds wait because the transcript was not yet
    # stable/complete. None: neither silence-based commit path ran this turn
    # (e.g. stopped_by_api, max_duration_reached).
    endpoint_shortcut_fired: bool | None = None
    # Which voice-activity detector actually produced this turn's speech
    # framing: "silero" or "rms". Silero is enabled by default but falls back
    # to RMS whenever the ONNX model file or onnxruntime is unavailable, or
    # the session's sample rate is one Silero does not support. That fallback
    # changes endpointing behaviour, so a turn trace that does not name the
    # detector cannot be compared against another run — CI's Tier 1 tests, for
    # instance, run before the model download and therefore exercise RMS.
    vad_backend: str | None = None


@dataclass
class TurnTrace:
    turn_id: str
    conversation_id: str
    started_at: str       # ISO8601 UTC
    ended_at: str | None
    wall: WallTimes = field(default_factory=WallTimes)
    asr: AsrSpan = field(default_factory=AsrSpan)
    agent: AgentSpan = field(default_factory=AgentSpan)
    tts: TtsSpan = field(default_factory=TtsSpan)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipelineLatencyStore:
    """Thread-safe ring buffer of the last ``maxlen`` TurnTrace records."""

    def __init__(self, maxlen: int = 20) -> None:
        self._lock = threading.Lock()
        self._buffer: deque[TurnTrace] = deque(maxlen=maxlen)

    def record(self, trace: TurnTrace) -> None:
        with self._lock:
            self._buffer.append(trace)

    def latest(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1].to_dict()

    def recent(self, n: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._buffer)
        return [t.to_dict() for t in items[-n:]]


# Module-level singleton — imported by audio_session and main.py
pipeline_store = PipelineLatencyStore()
