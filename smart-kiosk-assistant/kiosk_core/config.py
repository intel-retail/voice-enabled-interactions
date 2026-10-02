import os
from pathlib import Path


DEFAULT_ANALYZER_URL = os.getenv(
    "KIOSK_CORE_ANALYZER_URL",
    "http://127.0.0.1:8010/v1/audio/transcriptions",
)
DEFAULT_RAG_URL = os.getenv(
    "KIOSK_CORE_RAG_URL",
    "http://127.0.0.1:8020/api/v1/query",
)
DEFAULT_TTS_URL = os.getenv(
    "KIOSK_CORE_TTS_URL",
    "http://127.0.0.1:8011/v1/audio/speech",
)
# ── Continuous ASR streaming (WebSocket to audio-analyzer's /v1/realtime) ────
#
# When enabled, kiosk-core opens one persistent WebSocket per audio session
# and streams PCM frames to the analyzer continuously as they're captured,
# instead of POSTing a WAV file per flush. ASR then runs throughout speech
# (not just after a pause begins), so a fresh transcript is usually already
# available the instant the endpoint/silence decision fires -- this is what
# activates the otherwise-inert DEFAULT_ENDPOINT_SHORT_SECONDS shortcut above.
#
# This is a single global flag (not a per-session/per-request option) so the
# same kiosk-ui and benchmark REST APIs exercise whichever mode is selected
# server-side, with zero client-side changes either way. The file-based POST
# path (AnalyzerClient) remains untouched as the fallback: if the socket
# fails to connect for a given session, that session transparently falls
# back to the file-based path rather than failing the turn.
DEFAULT_ANALYZER_STREAMING_ENABLED = os.getenv(
    "KIOSK_CORE_ANALYZER_STREAMING_ENABLED", "false"
).lower() not in ("false", "0", "no")

# Cadence of non-blocking "preview" commits while streaming is active. Each
# commit only tells the analyzer "transcribe what's buffered so far" -- it
# never blocks the frame-capture loop. Matches the kiosk-voice-lab reference
# design's 0.4s rolling-snapshot cadence.
DEFAULT_REALTIME_PREVIEW_COMMIT_SECONDS = float(
    os.getenv("KIOSK_CORE_REALTIME_PREVIEW_COMMIT_SECONDS", "0.4")
)

# Faster preview-ping cadence used ONLY once trailing silence has begun
# (silence_run_seconds > 0), mirroring kiosk-voice-lab-main's
# tick_quiet_s=0.15 (pipeline/config.py): orchestrator.py's live loop switches
# `tick_s = 0.15 if in_silence else self.TICK_S` the instant its endpointer
# detects quiet, so a fresh transcript snapshot is almost always available
# well within endpoint_short_ms (150ms) of the customer's last word.
#
# Without this, VEI kept the flat 0.4s cadence during silence too, so
# whether the DEFAULT_ENDPOINT_SHORT_SECONDS completeness shortcut got a
# chance to fire depended on the luck of where in the 0.4s cycle speech
# happened to stop -- measured firing as low as 1/5 turns. Matching the
# lab's quiet-mode acceleration here removes that timing race.
DEFAULT_REALTIME_PREVIEW_COMMIT_QUIET_SECONDS = float(
    os.getenv("KIOSK_CORE_REALTIME_PREVIEW_COMMIT_QUIET_SECONDS", "0.15")
)

# Bounded wait for the FINAL commit of a turn only. If the analyzer's
# completion event doesn't land within this timeout, kiosk-core proceeds
# with whatever transcript snapshot is already available rather than
# hanging the turn.
DEFAULT_REALTIME_FINAL_COMMIT_TIMEOUT_SECONDS = float(
    os.getenv("KIOSK_CORE_REALTIME_FINAL_COMMIT_TIMEOUT_SECONDS", "2.5")
)

# Bounded wait for the initial WebSocket handshake/session.update round trip
# when opening a streaming session.
DEFAULT_REALTIME_CONNECT_TIMEOUT_SECONDS = float(
    os.getenv("KIOSK_CORE_REALTIME_CONNECT_TIMEOUT_SECONDS", "5.0")
)

DEFAULT_TTS_MODEL = os.getenv("KIOSK_CORE_TTS_MODEL", "qwen-tts")
DEFAULT_TTS_VOICE = os.getenv("KIOSK_CORE_TTS_VOICE")
DEFAULT_TTS_LANGUAGE = os.getenv("KIOSK_CORE_TTS_LANGUAGE", "English")
# ASR language hint sent with every transcription request. Left unset, Whisper
# auto-detects per 5-second chunk and regularly mis-fires on short kiosk
# utterances — measured over 25 real session chunks it produced Spanish output
# ("¿No puedo abrir el restaurante...") and one runaway repetition loop.
# Forcing "en" cut word error rate from 11.0% to 7.0% on CPU, GPU and NPU
# alike. Set to an empty string to restore auto-detection.
DEFAULT_ASR_LANGUAGE = os.getenv("KIOSK_CORE_ASR_LANGUAGE", "en") or None
# Whisper prompt injected with every transcription request. Primes the model
# with restaurant/menu vocabulary so it prefers domain-specific spellings over
# phonetically similar common words (e.g. "Classic Chicken Burger" over
# "classic senses price", "Do you serve burgers" over "Are you sir burgers?").
# The prompt is NOT instruction text — Whisper treats it as prior transcript
# context, so it should read like natural speech, not a list.
#
# Built from the live product catalogue rather than hardcoded: a hand-written
# list silently drifts from the menu. It previously omitted every
# Indian-origin item, which are exactly the names an English-forced Whisper
# mishears — "Aloo Tikki Burger" was transcribed as "and 2,000" and the item
# never reached the agent.
#
# Whisper's prompt window is 224 tokens; the catalogue is well inside that.
# Falls back to a static list when the YAML is unreadable (e.g. CI) so ASR is
# never broken by a missing seed file.
_ASR_PROMPT_FALLBACK = (
    "QuickBite Express restaurant. Classic Chicken Burger, Classic French Fries,"
    " Margherita Pizza, Mango Lassi, Cold Coffee, Fresh Lime Soda, Pepsi, 7UP."
    " Peri Peri Fries, Chocolate Lava Cake."
    " Menu, order, remove, add, confirm, cancel."
)


def _build_asr_prompt() -> str:
    """Compose the Whisper priming prompt from the product catalogue.

    Returns:
        Natural-language prompt naming every menu item, or a static fallback
        when the catalogue cannot be read.
    """
    try:
        import yaml  # local import: keeps config importable without PyYAML

        path = os.getenv(
            "KIOSK_CORE_PRODUCTS_YAML", "./configs/ordering/products.yaml"
        )
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        raw = data.get("products") if isinstance(data, dict) else data
        names: list[str] = []
        for product in raw or []:
            name = str(product.get("name", "")).strip()
            if not name:
                continue
            # Drop size/qty parentheticals ("(330 ml)", "(Regular)") — they are
            # packaging metadata, not words the customer says.
            name = name.split("(")[0].strip()
            if name and name not in names:
                names.append(name)
        if not names:
            return _ASR_PROMPT_FALLBACK
        return (
            "QuickBite Express restaurant. "
            + ", ".join(names)
            + ". Menu, order, remove, add, confirm, cancel."
        )
    except Exception:
        return _ASR_PROMPT_FALLBACK


DEFAULT_ASR_PROMPT = os.getenv("KIOSK_CORE_ASR_PROMPT") or _build_asr_prompt()
DEFAULT_TTS_INSTRUCTIONS = os.getenv("KIOSK_CORE_TTS_INSTRUCTIONS")
DEFAULT_SAMPLE_RATE = int(os.getenv("KIOSK_CORE_SAMPLE_RATE", "16000"))

# Audio capture source override.
# When HOST_MIC=true the backend captures audio directly from the host machine's
# microphone; otherwise the browser captures audio and streams it to the backend.
# This lets the same build work both locally (host mic) and against a
# remote/headless kiosk-core (browser mic) without auto-detection surprises.
HOST_MIC = os.getenv("HOST_MIC", "false").lower() not in ("false", "0", "no")

# ---------------------------------------------------------------------------
# TTS segment silence trimming
# ---------------------------------------------------------------------------
# The response is split at clause boundaries (",", ".", "!") so synthesis can
# start streaming after the first fragment — measured time-to-first-audio is
# 387 ms this way versus 1769 ms when splitting on sentences only, so the split
# is worth keeping. The cost is that every fragment arrives with its own
# leading/trailing silence baked in by the TTS model: across 10 real production
# segments that was 1.76 s of dead air (8% of the reply), heard as a long pause
# at every comma.
#
# Trimming each segment back to a deliberate pad turns that accidental pause
# into a controlled one. The pad is intentionally non-zero — clauses should
# breathe — but short enough to sound continuous.
DEFAULT_TTS_TRIM_ENABLED = os.getenv("KIOSK_CORE_TTS_TRIM_ENABLED", "true").lower() not in ("false", "0", "no")

# ── TTS worker concurrency (sentence-level parallel synthesis) ─────────────
# A multi-sentence reply is split into clauses/sentences and queued to
# _tts_worker one at a time. Historically only ONE worker thread drained that
# queue, so a 3-sentence reply paid 3 full synthesis round-trips back-to-back
# even though the text-to-speech service itself now runs multiple worker
# PROCESSES (see TEXT_TO_SPEECH_WORKERS in docker-compose.yml) and can serve
# concurrent requests. Measured on a 3-sentence upsell reply: serial
# ~2.9s total vs an estimated ~1.7-1.8s with 2-way parallelism.
# Safe to parallelize: downstream consumers (kiosk_core/service.py
# get_response_audio_path) look up a segment by its "index" field, not by
# the order it was appended to tts_audio_segments, so segments completing
# out of order (e.g. a short sentence 2 finishing before a long sentence 1)
# do not affect playback ordering. TtsClient's httpx.Client is documented
# thread-safe for concurrent calls (see tts_client.py).
# Does not change voice_to_voice_ms (sentence 1 is still queued/synthesised
# first either way) — this shortens total TURN completion time, i.e. less
# dead air between sentences 2/3 for multi-sentence replies.
# Default matches TEXT_TO_SPEECH_WORKERS (2) so kiosk-core never sends more
# concurrent requests than the backend has workers to serve without queueing.
DEFAULT_TTS_WORKER_CONCURRENCY = max(
    1, int(os.getenv("KIOSK_CORE_TTS_WORKER_CONCURRENCY", "2"))
)

# ── Speech normalization ───────────────────────────────────────────────────
# The reply text/UI keep "₹169" and "8 AM" as written; speecht5 reads those
# literally (symbol-and-digit tokens) rather than as a spoken price/time.
# Adapted from the reference prototype's pipeline/speech.py (kiosk-voice-lab
# -main), with ₹/Rs/INR support added for this restaurant's rupee prices.
# Deterministic regex/string work, no model involved — negligible latency
# cost. Applied inside TtsClient.synthesize_to_file so every call path
# (streamed clauses, the pre-synthesized opener) is covered.
DEFAULT_TTS_SPEECH_NORMALIZE_ENABLED = os.getenv(
    "KIOSK_CORE_TTS_SPEECH_NORMALIZE_ENABLED", "true"
).lower() not in ("false", "0", "no")

# ── Pre-synthesized opener ────────────────────────────────────────────────
# On an ordering turn the model emits only a tool call — no prose — and the
# spoken reply is templated from the tool result afterwards. Measured on this
# stack: the 33-token tool-call JSON costs ~1.6 s at ~48.5 ms/token, so the
# customer hears nothing for ~2 s after they stop speaking.
#
# Nothing can shorten that window from inside the turn (the reply does not
# exist yet), but it can be *filled*: a short phrase is synthesised once,
# cached on disk, and played the instant the turn ends while the agent call is
# still in flight. Time-to-first-audio drops from ~2 s to a file copy.
#
# The text MUST stay non-committal. Speech cannot be recalled, and the opener
# is spoken before any tool has run — so it must never name an item, quantity,
# price, or outcome, otherwise it can contradict the menu/removal/confirm
# guards that rewrite the real reply when a tool fails.
DEFAULT_OPENER_ENABLED = os.getenv(
    "KIOSK_CORE_OPENER_ENABLED", "true"
).lower() not in ("false", "0", "no")
DEFAULT_OPENER_TEXT = os.getenv("KIOSK_CORE_OPENER_TEXT", "One moment.")
# Rendered opener cache. Synthesised once per (text, voice, language) and
# reused for every turn and every session, so TTS never sits on the hot path.
DEFAULT_OPENER_CACHE_DIR = os.getenv("KIOSK_CORE_OPENER_CACHE_DIR", "./storage/openers")

# When list_products is called with no category, return a per-category summary
# instead of every product. The catalogue is 26 items: reciting it costs ~19 s
# of LLM generation and ~40 s of speech, and no kiosk customer listens to a
# 26-item list. Prompt rules alone did not stop the model reciting it.
# Set false to restore the full listing.
DEFAULT_LIST_PRODUCTS_SUMMARY = os.getenv(
    "KIOSK_CORE_LIST_PRODUCTS_SUMMARY", "true"
).lower() not in ("false", "0", "no")
# Silence kept before speech starts, in every segment.
DEFAULT_TTS_LEAD_PAD_MS = float(os.getenv("KIOSK_CORE_TTS_LEAD_PAD_MS", "20"))
# Trailing silence for a fragment ending mid-sentence (",", ":", ";").
DEFAULT_TTS_CLAUSE_PAD_MS = float(os.getenv("KIOSK_CORE_TTS_CLAUSE_PAD_MS", "60"))
# Trailing silence for a fragment ending a sentence (".", "!", "?").
DEFAULT_TTS_SENTENCE_PAD_MS = float(os.getenv("KIOSK_CORE_TTS_SENTENCE_PAD_MS", "150"))
# Amplitude below this fraction of the segment peak counts as silence.
DEFAULT_TTS_SILENCE_FLOOR = float(os.getenv("KIOSK_CORE_TTS_SILENCE_FLOOR", "0.02"))
# Word cap applied to the FIRST spoken segment of a turn only.
#
# Measured Kokoro synthesis cost is ~248ms fixed + ~13.9ms per character, so
# the first segment's length translates almost linearly into time-to-first-
# audio. A typical order confirmation ("I've added Classic Chicken Burger to
# your order.", 47 chars) costs ~900ms before the customer hears anything,
# even though the leading noun phrase alone already carries the confirmation.
#
# Capping only the first segment lets that leading phrase reach the
# synthesizer immediately; the remainder becomes segment 2 and is synthesized
# while segment 1 is already playing, so it costs nothing on the critical
# path. Later segments are deliberately NOT capped -- they are already fully
# overlapped with playback and splitting them would only add per-segment
# fixed cost (248ms each) and extra prosody seams.
#
# 5 words is chosen to land after the object noun phrase rather than inside
# it ("I've added Classic Chicken Burger" / "to your order."). Set to 0 to
# disable the split entirely.
DEFAULT_TTS_FIRST_PHRASE_MAX_WORDS = int(
    os.getenv("KIOSK_CORE_TTS_FIRST_PHRASE_MAX_WORDS", "5")
)
# Minimum number of words that must remain AFTER the cap for a split to be
# worthwhile. Without this, a 6-word sentence would be chopped into a 5-word
# head and a 1-word orphan: the orphan pays the full ~248ms fixed synthesis
# cost and introduces an audible seam, while saving almost no time on the
# first segment.
DEFAULT_TTS_FIRST_PHRASE_MIN_TAIL_WORDS = int(
    os.getenv("KIOSK_CORE_TTS_FIRST_PHRASE_MIN_TAIL_WORDS", "3")
)
# Linear fade-in/fade-out applied to the very edges of every trimmed segment,
# in milliseconds. The trim cut lands on whatever raw sample index the pad
# window computes to — not a zero-crossing — so the waveform can (and does,
# depending on the phoneme at that exact point) have a non-zero value right
# at the edge. The UI schedules segments back-to-back on one Web Audio
# timeline with no gap and no crossfade (see useAudioQueue.ts), so any such
# jump between one segment's last sample and the next segment's first sample
# is heard as an audible click — intermittently, only on the sentences whose
# cut points happen to land off a zero-crossing. This ramps every segment's
# edges to true zero unconditionally, which fully removes that click
# regardless of content. Kept well inside the lead/clause/sentence pads above
# so it never touches actual speech.
DEFAULT_TTS_FADE_MS = float(os.getenv("KIOSK_CORE_TTS_FADE_MS", "5"))

# TTS output loudness. SpeechT5's vocoder outputs a quiet, roughly constant
# level regardless of which speaker embedding is selected — swapping voices
# (e.g. Ryan -> Kabir) changes timbre, not level, so a soft-sounding kiosk is
# a gain problem, not a voice problem. There is no gain/volume control in the
# text-to-speech service itself, so kiosk-core normalizes + boosts every
# synthesized segment in place before playback.
#
# Two knobs stack: first each segment's peak is normalized up (or down) to
# TARGET_PEAK, then an extra flat boost of GAIN_DB is applied on top. Total
# applied gain is hard-clamped at GAIN_MAX_DB so that a near-silent/failed
# synthesis (e.g. a clipped word) can't be amplified into harsh noise.
#
# TARGET_PEAK is intentionally kept a little below full scale (not 1.0):
# observed live, pushing peaks to 100% FS left zero headroom, so any
# downstream OS/mixer volume above unity (venue speaker volume nudged up on
# demo day, PipeWire sink gain >100%, etc.) clipped and was heard as
# crackling/distortion. 0.9 keeps ~1 dB of headroom against exactly that.
DEFAULT_TTS_GAIN_ENABLED = os.getenv("KIOSK_CORE_TTS_GAIN_ENABLED", "true").lower() not in ("false", "0", "no")
# Fraction of full-scale (int16 max) each segment's peak is normalized to.
DEFAULT_TTS_TARGET_PEAK = float(os.getenv("KIOSK_CORE_TTS_TARGET_PEAK", "0.9"))
# Extra flat boost in dB applied on top of peak normalization. Raise this
# first if the kiosk still sounds quiet over venue noise after normalization.
DEFAULT_TTS_GAIN_DB = float(os.getenv("KIOSK_CORE_TTS_GAIN_DB", "4.0"))
# Safety ceiling on total applied gain (normalization + boost combined).
DEFAULT_TTS_GAIN_MAX_DB = float(os.getenv("KIOSK_CORE_TTS_GAIN_MAX_DB", "15.0"))

# ── Opener TTS cache ────────────────────────────────────────────────────────
# Process-wide reuse of already-synthesised SHORT opener segments ("Got it.",
# "Sure.", "Of course.") across turns and sessions.
#
# Why this exists: measured on Kokoro/CPU, synthesising the single word-pair
# "Got it." costs ~270-300 ms, and that cost sits squarely on the critical
# path — it is the last stage before the customer hears anything, so it is
# ~27% of a ~1030 ms voice-to-voice turn. The agent opens the overwhelming
# majority of ordering turns with the same handful of stock phrases, so the
# same audio is paid for again on every single turn.
#
# How it differs from DEFAULT_SPECULATIVE_TTS_PRESYNTH_ENABLED (which is off
# by default, and should stay off): speculative pre-synthesis ISSUES EXTRA TTS
# REQUESTS ahead of the real turn, and because the TTS backend serialises
# requests those extras queue in front of the real reply and make the turn
# dramatically slower (see the 2026-09-08 note on the speculative flags).
# This cache never issues a request of its own. It only ever keeps a COPY of a
# segment the real pipeline already synthesised, fully trimmed and gain-
# adjusted, and replays that identical file. Worst case on a miss is the
# status quo; there is no path by which it adds load.
#
# Safety: an entry is only served when the normalized sentence text AND the
# full voice/model/language/instructions tuple match, so a cached clip can
# never be spoken for different text or in the wrong voice.
DEFAULT_TTS_OPENER_CACHE_ENABLED = os.getenv(
    "KIOSK_CORE_TTS_OPENER_CACHE_ENABLED", "true"
).lower() not in ("false", "0", "no")
# Only segments at or below this many characters are cached. Deliberately
# small: stock openers are short and highly repetitive, whereas the sentences
# that carry order details ("Your total is now ₹169.") are long, vary every
# turn, and would never be hit again — caching them would only burn disk.
DEFAULT_TTS_OPENER_CACHE_MAX_CHARS = int(
    os.getenv("KIOSK_CORE_TTS_OPENER_CACHE_MAX_CHARS", "24")
)
# Hard ceiling on retained entries, so an unexpectedly chatty model cannot
# grow the cache without bound over a long kiosk uptime. Once full, the cache
# simply stops admitting new phrases (the common openers are learned within
# the first few turns, so eviction churn buys nothing).
DEFAULT_TTS_OPENER_CACHE_MAX_ENTRIES = int(
    os.getenv("KIOSK_CORE_TTS_OPENER_CACHE_MAX_ENTRIES", "32")
)

# Metrics collector – base URL of the standalone metrics-collector container.
# Within Docker the service is reachable as http://metrics-collector:9000.
METRICS_COLLECTOR_URL = os.getenv(
    "KIOSK_CORE_METRICS_URL",
    "http://metrics-collector:9000",
)
# Hard cap on how much audio accumulates before a chunk is force-flushed.
#
# Raised 2.5 → 6.0s. At 2.5s this cap fired mid-utterance on almost every turn,
# because a typical kiosk request ("Can you suggest something to drink") is
# 2.5-4s of *continuous* speech with no pause long enough to trigger the
# adaptive flush. Two distinct defects followed from that:
#
#   1. ASR quality — the cut landed mid-word, so Whisper saw fragments.
#      Observed verbatim in production logs: "something" was split into
#      "Can you suggest some" + "thing to drink." This is the same failure the
#      adaptive_flush_pause 0.30 → 0.70 change fixed, arriving via a different
#      path.
#   2. Speaker identity — the severed tail is a 0.5-1.5s fragment. Speaker
#      embeddings computed over such a short span are unreliable, so the
#      analyzer clustered the tail as a *different* speaker and the
#      diarization filter discarded the customer's own words.
#
# 6.0s lets a normal single utterance complete inside one chunk. The adaptive
# pause flush (0.70s) still cuts at natural boundaries, so latency for ordinary
# speech is unchanged — this cap now only bites on genuinely continuous speech,
# where a longer chunk also yields better ASR context and a more reliable
# speaker embedding.
DEFAULT_CHUNK_SECONDS = float(os.getenv("KIOSK_CORE_CHUNK_SECONDS", "6.0"))
# Trailing silence that ends a turn.
#
# INVARIANT: must be strictly greater than DEFAULT_ADAPTIVE_FLUSH_PAUSE_SECONDS.
# If it is not, the endpoint fires before the adaptive flush can ever run (its
# guard requires silence_run < silence_timeout) and the pre-warm optimisation
# below is silently dead. The kiosk-ui previously sent 0.65s while the flush
# pause was 0.70s, which is exactly what happened.
#
# Note the browser UI overrides this per session via silence_timeout_seconds
# (kiosk-ui/src/constants.ts), so that constant governs the kiosk; this default
# applies to microphone and file sessions.
#
# 1.5s: at 0.65s a customer pausing to read the menu was cut off mid-sentence,
# so Whisper only ever received 1.8-2.5s of truncated audio and had to guess
# the item name. A genuine end-of-turn pause is ~1.0-1.5s, so this tolerates
# hesitation while keeping the reply prompt.
#
# 1.1s (kiosk-voice-lab-main parity): the lab's SmartEndpointer waits 1.1s for
# an "unfinished" turn (endpoint_long_ms). Lowered from 1.5s to match. This is
# the ceiling every turn without an early completeness commit still pays, so
# it inherits the same mid-sentence-hesitation risk the 1.5s value was chosen
# to avoid — re-validate against the fixture benchmark before trusting this
# in the live demo (see docs/performance-improvements-2026-09.md).
DEFAULT_SILENCE_TIMEOUT_SECONDS = float(os.getenv("KIOSK_CORE_SILENCE_TIMEOUT_SECONDS", "1.1"))
# Adaptive mid-utterance flush: when silence reaches this threshold but hasn't
# yet hit silence_timeout_seconds, flush the accumulated chunk to the background
# ASR worker so processing starts immediately. The tail chunk at true endpoint
# will then be short (only the frames since the last adaptive flush), cutting
# critical-path ASR from up to chunk_seconds down to ~0.3-0.5s of audio.
# Raised from 0.30 → 0.70s: at 0.30s, natural in-phrase pauses (e.g. between
# "chicken" and "burger") triggered an adaptive flush mid-word, clearing
# chunk_frames so the next word had no sentence context — Whisper then
# hallucinated ("chip" for "chicken") or misread the isolated tail ("Kin Burger"
# for "burger").
#
# Lowered 0.70 -> 0.50s (2026-09-10, ITEP endpoint-latency work): re-tested the
# boundary with tests/benchmarks/v2v_fixture_benchmark.py against rec1_16k.wav
# and rec2_16k.wav (real recorded speech, 3 runs each):
#   * 0.50s: transcripts identical/correct vs the 0.70s baseline on both
#     fixtures, 6/6 runs. endpoint_wait_ms dropped ~150ms (rec1: ~1,000ms ->
#     ~900ms). SAFE.
#   * 0.40s: reproduced the same class of hallucination the 0.30s value
#     caused originally — "Good." became "Good, good, good." (repeated word)
#     and a phantom "Bye, bye." was inserted; rec2's trailing "french fry" was
#     truncated to "one Fr". REGRESSION, do not use.
# 0.50s is therefore the validated floor for this ASR model/hardware, not a
# guess — do not lower further without re-running that benchmark and manually
# inspecting transcripts for repeated/phantom words the way this note did.
# Kept above DEFAULT_ENDPOINT_SHORT_SECONDS's inert threshold — see the
# INVARIANT note above (that check still can't fire before this flush lands).
DEFAULT_ADAPTIVE_FLUSH_PAUSE_SECONDS = float(os.getenv("KIOSK_CORE_ADAPTIVE_FLUSH_PAUSE_SECONDS", "0.50"))
# ── Preview flush (continuous mid-speech ASR, not just at the pause) ───────
# Without this, a long utterance is only ever flushed to ASR once — at the
# adaptive pause above, AFTER the customer stops talking — so the entire
# utterance's audio is on the critical path for that one ASR round-trip. That
# round-trip has been measured landing at ~1.23-1.28s, later than even
# DEFAULT_ENDPOINT_SHORT_SECONDS (1.0s), which is why the completeness
# shortcut below rarely gets a chance to fire (see
# docs/performance-improvements-2026-09.md, "preview-ASR work").
#
# This flushes accumulated speech to the background ASR worker periodically
# WHILE the customer is still talking (not just at chunk_seconds or the
# pause), so by the time silence begins, most of the utterance has already
# been transcribed off the critical path — the chunk flushed at the adaptive
# pause is then much shorter (only the audio since the last preview tick),
# lands sooner, and gives the completeness shortcut a real chance to fire.
#
# Every preview flush reuses the exact same _flush_chunk path as the
# existing chunk-size-cap and adaptive-pause flushes — same
# transcript_parts append, same DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED
# gating, same >=0.5s minimum-content guard. It is not a separate throwaway
# code path; it only changes WHEN a chunk boundary is cut.
#
# Gated by self._flush_queue.unfinished_tasks == 0 at the call site (i.e. no
# other flush is currently in flight) because audio-analyzer serialises all
# WhisperPipeline.generate() calls behind a single global lock — a shorter
# fixed tick interval would just queue up requests behind an already-running
# call rather than run concurrently.
#
# 1.5s default: long enough that it never fires on typical short utterances
# ("one classic burger" ~1-2s) — behaviour there is unchanged from before.
# It only engages on longer continuous speech, which is exactly where the
# single end-of-utterance flush was most expensive.
DEFAULT_PREVIEW_FLUSH_ENABLED = os.getenv("KIOSK_CORE_PREVIEW_FLUSH_ENABLED", "true").lower() == "true"
# 0.4s (kiosk-voice-lab-main parity, change #1 "continuous listening"): lowered
# from 1.5s. This is a TRIGGER threshold, not a guaranteed cadence — the loop
# only fires the next preview flush once flush_queue.unfinished_tasks == 0, so
# real cadence is bounded below by however long the previous ASR call took to
# come back (whisper-small/NPU: ~500-550ms fixed cost per call, per
# configs/audio-analyzer/config.yaml), not by this constant. Lowering it to
# 0.4s just removes the ARTIFICIAL 1.5s gap on top of that natural floor, so
# transcripts arrive roughly every ASR-round-trip instead of every 1.5s.
#
# NOTE: full kiosk-voice-lab-main parity (distil-whisper distil-small.en on
# NPU, ~40-70ms/snapshot) is NOT used here — that model has a confirmed
# state-corruption bug on this NPU + openvino_genai stack when the same
# pipeline instance processes varying clip lengths back-to-back (exactly what
# a growing rolling snapshot does). See configs/audio-analyzer/config.yaml's
# models.asr.name comment. whisper-small does not corrupt under the same
# conditions (verified), so it stays the model here despite being slower.
DEFAULT_PREVIEW_FLUSH_INTERVAL_SECONDS = float(os.getenv("KIOSK_CORE_PREVIEW_FLUSH_INTERVAL_SECONDS", "1.5"))

# ── Speculative drafting (Round 3: agent/TTS cache-warming ahead of endpoint) ─
#
# The ASR-side fixes above (persistent httpx client, preview-ASR flush)
# close the gap on transcription only. The dominant remaining cost by far is
# the agent/LLM call + TTS synthesis, which today only ever starts AFTER the
# customer stops talking and the endpoint fires — this is the structural
# reason the reference "lab" prototype reaches ~700ms voice-to-voice while
# this pipeline sits several seconds higher: the lab fires a fresh
# agent+TTS draft on every ASR tick DURING speech (newest-wins), so a
# matching draft/audio usually already exists by the time its endpoint
# fires; here, nothing downstream of ASR has reacted to a preview transcript
# until now.
#
# This flag enables that: on every preview-ASR flush (see
# DEFAULT_PREVIEW_FLUSH_ENABLED above), a background thread fires a
# SPECULATIVE agent turn (chat(..., speculative=True)) against the
# transcript-so-far. The server FORCES every mutating ordering tool
# (place_order/update_order/confirm_.../cancel_order/remove_from_order) into
# dry_run mode for the whole turn — see _MUTATING_TOOLS/_speculative_ctx in
# plugins/kiosk/ordering_agent.py — so a speculative call built on a
# still-changing, possibly-incomplete transcript can NEVER write a real row
# to the orders database, no matter what the agent decides to call. This
# was verified directly against a live SQLite file (dry_run=True leaves order/
# order_item row counts unchanged) before this flag was wired in.
#
# "Newest-wins": only the LATEST-STARTED draft's result is ever kept (see
# BaseAudioSession._speculative_generation) — an older call that happens to
# return after a newer one started is discarded outright, since a longer or
# corrected transcript snapshot makes it stale.
#
# Kept deliberately SAFE-SCOPED for this rollout: the real (endpoint-fired)
# turn always still runs its own full, fully-guarded LLM call exactly as
# before — nothing here skips or replaces it. The benefit is (a) the LLM's
# own prefix-cache/tool-call code path is already warm by the time the real
# call runs, and (b) DEFAULT_SPECULATIVE_TTS_PRESYNTH_ENABLED below can
# pre-synthesise the draft's predicted reply so a byte-identical real
# sentence is served instantly instead of re-synthesised. A true "skip the
# LLM/guards entirely and replay the draft's tool calls for real" optimisation
# (closing the rest of the gap to the lab's ~700ms) is intentionally NOT
# implemented here — it would require replaying draft results through the
# same guard/reply-construction pipeline the LLM path uses, not just
# re-dispatching raw tool calls, and was judged too large/risky to rush
# alongside everything else in this round. Recommended as a dedicated,
# carefully-tested follow-up.
# DEFAULTED TO FALSE (2026-09-08 live-replay finding): both the LLM (OVMS)
# and TTS backends in this deployment serialise requests — they do not run
# a speculative call and the real turn's call concurrently, they queue them.
# Live replay of tests/rec2_16k.wav showed the real turn's own tts= time
# balloon to ~21.8s, matching almost exactly the ~21s consumed synthesising
# 5 speculative sentences immediately beforehand — i.e. the real turn was
# stuck waiting behind its own speculative work, not helped by it. Overall
# wall time for that turn (38.2s) was far WORSE than the pre-speculative
# baseline (~5.8s). Until the backends are confirmed/configured to serve
# concurrent requests (e.g. multiple OVMS/TTS worker instances or a request
# queue with priority for the real turn), enabling this trades a latency
# win for a latency loss. Code path is fully safety-verified (see dry_run
# notes above) and left in place as an opt-in for when backend concurrency
# is addressed — do not flip to "true" without re-validating via a live
# replay first.
DEFAULT_SPECULATIVE_DRAFT_ENABLED = os.getenv("KIOSK_CORE_SPECULATIVE_DRAFT_ENABLED", "false").lower() == "true"

# Pre-synthesise the speculative draft's predicted reply sentence-by-sentence
# during speech, cached by exact sentence text. The REAL turn's _tts_worker
# checks this cache before calling TTS for a sentence — only a cache HIT is
# ever served, and a hit is only possible when the real, fully-guarded reply
# produces a sentence identical to what the draft predicted, so nothing is
# ever spoken that the real pipeline didn't independently decide to say.
# Defaulted to false for the same backend-serialisation reason as
# DEFAULT_SPECULATIVE_DRAFT_ENABLED above — see that comment.
DEFAULT_SPECULATIVE_TTS_PRESYNTH_ENABLED = (
    os.getenv("KIOSK_CORE_SPECULATIVE_TTS_PRESYNTH_ENABLED", "false").lower() == "true"
)
# ── Adaptive endpoint (sentence-completeness shortcut) ──────────────────────
# DEFAULT_SILENCE_TIMEOUT_SECONDS above has to be long enough for the WORST
# case: a customer hesitating mid-sentence. That makes every turn pay the
# hesitation tax, including the majority that end on an obviously finished
# sentence ("I would like one classic chicken burger").
#
# A loudness detector cannot tell those two apart, so it needs one fixed wait.
# Reading the transcript can: if the words so far end on a filler ("um"), a
# dangling function word ("and", "with", "I'd") or a comma, the customer is
# mid-thought and we keep the full timeout. Otherwise we commit early.
#
# This only ever SHORTENS the wait, and only when a transcript already exists.
# If ASR has not returned yet, the text is empty or stale, the check fails
# closed and behaviour is identical to the fixed timeout.
#
# INVARIANT (WAIVED, see below): the guard normally requires this to be
# strictly greater than DEFAULT_ADAPTIVE_FLUSH_PAUSE_SECONDS (0.50) and
# strictly less than DEFAULT_SILENCE_TIMEOUT_SECONDS (1.1), so the adaptive
# flush has fired and its ASR result has had time to land before the
# completeness check runs.
#
# 0.15s (kiosk-voice-lab-main parity, endpoint_short_ms): matches the lab's
# value for a transcript that already reads as finished. KNOWN LIMITATION:
# on THIS pipeline the value is inert at 0.15s, because the adaptive flush
# that produces the transcript the completeness check reads does not fire
# until 0.50s of silence (DEFAULT_ADAPTIVE_FLUSH_PAUSE_SECONDS) — the
# `_adaptive_flushed` guard in audio_session.py is still False at 0.15s, so
# every turn falls through to the DEFAULT_SILENCE_TIMEOUT_SECONDS wait
# regardless of this setting. The lab reaches 0.15s because its NPU runs
# continuous ASR (a rolling transcript snapshot every ~0.4s WHILE the
# customer is still talking — see kiosk-voice-handoff.pdf, change #1), so a
# fresh transcript is always available well before 150ms of silence. Set to
# 0.15 here as a forward-looking value; it becomes load-bearing once
# continuous/rolling ASR replaces the flush-on-pause design (tracked as the
# lab's change #1: "put listening on the NPU, and do it continuously").
DEFAULT_ENDPOINT_COMPLETE_ENABLED = os.getenv("KIOSK_CORE_ENDPOINT_COMPLETE_ENABLED", "true").lower() == "true"
DEFAULT_ENDPOINT_SHORT_SECONDS = float(os.getenv("KIOSK_CORE_ENDPOINT_SHORT_SECONDS", "0.15"))
# Minimum words before a transcript may be judged "finished". Two words or
# fewer is almost always a fragment mid-utterance ("I want...").
DEFAULT_ENDPOINT_MIN_WORDS = int(os.getenv("KIOSK_CORE_ENDPOINT_MIN_WORDS", "3"))
# Stability window for the completeness shortcut: the transcript text must be
# UNCHANGED for this long before "reads complete" is trusted, not just true on
# the instant a chunk lands.
#
# Reference: kiosk-voice-lab-main's assess_complete(text, stable) requires two
# consecutive tick snapshots to agree before treating a transcript as finished
# — its own comment notes "a word list alone called 90% of fragments
# 'finished'". _looks_complete() here is the same kind of word-list check, so
# it inherits the same risk: a transcript that happens to end on a real word
# the instant a chunk lands (e.g. a flush landing mid-utterance with
# "...order one classic chicken" before "burger" has even been spoken) could
# otherwise commit the turn on a truncated sentence.
#
# This only ever WITHHOLDS an early commit, never grants one the fixed timeout
# wouldn't have — if the transcript is still changing, the full
# silence_timeout_seconds wait is the fallback, same fail-closed posture as
# _looks_complete's own empty-transcript case.
#
# Lowered from 0.2 to 0 (2026-09) after measuring the shortcut's actual firing
# rate in streaming mode: the two-confirmation design needs TWO preview round
# trips (each ~90-220ms, see RealtimeAnalyzerClient) to land inside the
# ~0.15-1.1s shortcut window, which the quiet-mode preview cadence
# (DEFAULT_REALTIME_PREVIEW_COMMIT_QUIET_SECONDS=0.15s) only barely allows —
# measured firing rate 1/4 turns (25%) in a live 4-turn conversation. 0
# (trust the FIRST snapshot that reads complete, per this constant's own
# single-confirmation mode above) raised that to a consistent 3-4/4 turns
# (75-100%) across repeated full-conversation runs (real endpoint detection)
# AND 7/10 turns across 5 repeats of tests/benchmarks/v2v_fixture_benchmark.py
# (real recorded speech, not synthesised) — every transcript in all of these
# runs stayed accurate/untruncated, so the truncated-mid-utterance risk this
# constant guards against was not observed in practice for this pipeline's
# ASR latency profile. Re-validate with the same fixture benchmark (checking
# transcripts for truncation/hallucination, not just the firing rate) before
# raising this back above 0.
#
# Trade-off observed at 0, NOT eliminated: firing this early sometimes beats
# DEFAULT_ADAPTIVE_FLUSH_PAUSE_SECONDS's own real (destructive) commit to the
# punch, so that commit hasn't cleared the buffer yet and the final flush has
# to transcribe the whole utterance instead of a short tail — measured
# final_flush_wait_ms ~420-560ms on turns where endpoint_wait_ms landed at the
# ~200ms floor, vs. ~10-70ms on turns where it landed >=600ms (adaptive commit
# already won). Net effect across 3 full-conversation runs was still a clear
# win end to end (endpoint_wait_ms + final_flush_wait_ms combined dropped
# from ~1200-1700ms/turn to ~600-750ms/turn on the turns that fire), but this
# is why total v2v didn't fall as far as the raw firing-rate jump alone would
# suggest — see docs/performance-improvements-2026-09.md before tuning either
# constant further.
DEFAULT_ENDPOINT_STABLE_SECONDS = float(os.getenv("KIOSK_CORE_ENDPOINT_STABLE_SECONDS", "0"))
DEFAULT_MAX_SESSION_SECONDS = float(os.getenv("KIOSK_CORE_MAX_SESSION_SECONDS", "20.0"))
DEFAULT_SILENCE_THRESHOLD = int(os.getenv("KIOSK_CORE_SILENCE_THRESHOLD", "900"))

# ── ASR trailing-silence trim (kiosk-voice-lab-main parity) ──────────────────
# Whisper hallucinates a sentence-completing word/phrase when fed audio that
# ends in "dangling speech + silence" — e.g. "...one classic chicken" + a
# beat of quiet gets decoded as "...one classic chicken burger. Good." This
# is the source of the spurious trailing tokens (measured: "Good.") that
# break EXACT-STRING speculative-draft matching (see
# DEFAULT_SPECULATIVE_DRAFT_ENABLED) — the final transcript rarely matches
# byte-for-byte any earlier preview-transcript snapshot, so drafts almost
# never hit.
#
# kiosk-voice-lab-main's fix (pipeline/orchestrator.py, AdaptivePipeline):
# trim the audio actually SENT to ASR down to
# "last detected speech sample + a short decay tail", discarding the rest of
# the accumulated silence. This changes only what ASR is asked to transcribe
# — it does NOT touch endpoint/silence-timeout timing, which keeps counting
# the full silence_run_seconds exactly as before.
#
# Feature-flagged OFF by default: needs a live A/B (transcript accuracy,
# hallucination rate, and — the actual payoff — speculative-draft hit rate)
# before being trusted in production. See docs/performance-improvements-
# 2026-09.md for the kind of validation this class of change gets before
# being flipped on.
DEFAULT_ASR_TRIM_TRAILING_SILENCE_ENABLED = (
    os.getenv("KIOSK_CORE_ASR_TRIM_TRAILING_SILENCE_ENABLED", "false").lower() == "true"
)
# How much trailing silence to KEEP (the "decay tail") when trimming — mirrors
# kiosk-voice-lab-main's 0.15s exactly (speech_end_samples + int(0.15 * sr)).
# Too short risks clipping genuine trailing speech if VAD's silence-onset
# detection lags by a frame or two; too long re-invites the hallucination the
# trim exists to prevent.
DEFAULT_ASR_TRIM_DECAY_SECONDS = float(os.getenv("KIOSK_CORE_ASR_TRIM_DECAY_SECONDS", "0.15"))


# ── Adaptive VAD (noise-floor calibration) ────────────────────────────────────
# DEFAULT_SILENCE_THRESHOLD is an absolute int16 RMS value, which is only ever
# correct for the microphone and room it was measured on. Measured on the demo
# unit (PCM2902 USB codec, quiet room, nobody speaking) the *silence* floor was
# RMS ~1076 — i.e. above the 900 gate, so every frame classified as speech,
# silence_run_seconds never accumulated, and neither the silence endpoint nor
# the adaptive flush could ever fire. A louder venue makes that worse.
#
# Rather than hand-tuning the constant per venue (impossible when the venue
# cannot be tested beforehand), the session measures the actual noise floor at
# runtime and places the speech gate a fixed margin above it.
#
# FAIL-OPEN BY CONSTRUCTION: the derived gate is clamped to
# [THRESHOLD_MIN, THRESHOLD_MAX]. If a venue is so loud that floor*margin
# exceeds THRESHOLD_MAX, the gate saturates at THRESHOLD_MAX and behaviour
# degrades to "treat everything as speech" — exactly today's behaviour, never
# worse. A too-HIGH gate is the dangerous direction (speech is never detected
# and nothing is transcribed at all), which the ceiling exists to prevent.
ADAPTIVE_VAD_ENABLED = os.getenv("KIOSK_CORE_ADAPTIVE_VAD_ENABLED", "true").lower() not in ("false", "0", "no")
# Audio observed before the gate is derived. Frames in this window are held in
# the preroll buffer (which is sized to cover it), so nothing is lost.
DEFAULT_VAD_CALIBRATION_SECONDS = float(os.getenv("KIOSK_CORE_VAD_CALIBRATION_SECONDS", "0.5"))
# Low percentile of frame RMS used as the floor estimate. Deliberately low so
# that a customer who starts talking immediately (making some calibration frames
# loud) still yields a floor drawn from the quiet frames between words.
DEFAULT_VAD_FLOOR_PERCENTILE = float(os.getenv("KIOSK_CORE_VAD_FLOOR_PERCENTILE", "20"))
# How far above the measured floor the speech gate sits. 6 dB ~= 2x the floor.
# Measured against real speech mixed over the real PCM2902 noise floor: at 9 dB
# the gate needed ~15 dB SNR before it reliably saw speech, which a kiosk mic at
# arm's length will not deliver. 6 dB detects normal speech from ~9 dB SNR while
# still sitting clear of the floor.
DEFAULT_VAD_MARGIN_DB = float(os.getenv("KIOSK_CORE_VAD_MARGIN_DB", "6.0"))
DEFAULT_VAD_THRESHOLD_MIN = int(os.getenv("KIOSK_CORE_VAD_THRESHOLD_MIN", "300"))
DEFAULT_VAD_THRESHOLD_MAX = int(os.getenv("KIOSK_CORE_VAD_THRESHOLD_MAX", "4000"))
# Once calibrated, the floor keeps tracking on non-speech frames only (standard
# VAD practice — never adapt the noise estimate while speech is present, or the
# gate climbs during a long utterance and cuts the customer off).
#
# Adaptation is ASYMMETRIC. Quiet frames *within* an utterance (gaps between
# words) fall below the gate and are therefore seen as "non-speech"; with a
# symmetric EMA they dragged the floor upward mid-sentence — measured drift was
# 1098 -> 1642, which pushed the gate up and swallowed the rest of the
# utterance. Downward moves (room genuinely got quieter) are safe and track
# quickly; upward moves are deliberately ~10x slower.
DEFAULT_VAD_FLOOR_ADAPT_DOWN = float(os.getenv("KIOSK_CORE_VAD_FLOOR_ADAPT_DOWN", "0.05"))
DEFAULT_VAD_FLOOR_ADAPT_UP = float(os.getenv("KIOSK_CORE_VAD_FLOOR_ADAPT_UP", "0.005"))
DEFAULT_BLOCK_DURATION_SECONDS = float(os.getenv("KIOSK_CORE_BLOCK_DURATION_SECONDS", "0.1"))
DEFAULT_PREROLL_SECONDS = float(os.getenv("KIOSK_CORE_PREROLL_SECONDS", "0.3"))

# Silero VAD (v5, onnxruntime): a model-based alternative to the adaptive RMS
# VAD above. When enabled, Silero's speech probability REPLACES the RMS-derived
# is_speech decision in the per-frame loop (see
# BaseAudioSession._process_frame_stream); the RMS floor/gate calibration still
# runs alongside it (cheap, harmless) but its classification is ignored while
# Silero is active.
#
# Default flipped OFF → ON. Rationale: every downstream timer in the turn — the
# adaptive flush at DEFAULT_ADAPTIVE_FLUSH_PAUSE_SECONDS (0.50s), the
# sentence-completeness shortcut at DEFAULT_ENDPOINT_SHORT_SECONDS, and the
# DEFAULT_SILENCE_TIMEOUT_SECONDS endpoint — starts counting from the frame the
# VAD first calls silence. An RMS gate is an absolute loudness threshold, so in
# a room louder than the one it was calibrated for it declares silence LATE (or,
# as measured on the demo unit, never: the silence floor read RMS ~1076 against
# a 900 gate, so silence_run_seconds never accumulated and neither the endpoint
# nor the adaptive flush could fire at all). The adaptive floor calibration
# added later mitigates that but still tracks loudness, not speech. Silero
# scores speech directly, so the silence clock starts at the true end of the
# customer's words and every timer downstream shifts earlier with it.
#
# Cost is small and bounded: a ~2MB int8 ONNX graph pinned to one intra-op
# thread, run on 0.1s frames.
#
# Fails open — the constructor falls back to the RMS VAD if onnxruntime or the
# model file is unavailable, so this flag cannot break a session. Set
# KIOSK_CORE_SILERO_VAD_ENABLED=false to restore the RMS-only path.
KIOSK_CORE_SILERO_VAD_ENABLED = os.getenv("KIOSK_CORE_SILERO_VAD_ENABLED", "true").lower() not in (
    "false",
    "0",
    "no",
)
# Bundled with this repo (~2MB) rather than fetched at runtime, since
# kiosk-core is built fresh from source each deploy (unlike audio-analyzer/
# text-to-speech, which pull prebuilt images with their own model-fetch
# scripts).
DEFAULT_SILERO_VAD_MODEL_PATH = os.getenv(
    "KIOSK_CORE_SILERO_VAD_MODEL_PATH",
    str(Path(__file__).resolve().parent / "models" / "silero_vad.onnx"),
)
# Speech-probability cutoff. 0.5 matches the reference lab's SmartEndpointer
# threshold and Silero's own documented default.
DEFAULT_SILERO_VAD_THRESHOLD = float(os.getenv("KIOSK_CORE_SILERO_VAD_THRESHOLD", "0.5"))
# onnxruntime intra-op thread cap for the Silero session. Kept at 1 since the
# model is tiny (~2MB) and does not benefit from parallelism; avoids
# contending with the ASR/LLM/TTS pipelines' own thread pools.
DEFAULT_SILERO_VAD_INTRA_OP_THREADS = int(os.getenv("KIOSK_CORE_SILERO_VAD_INTRA_OP_THREADS", "1"))
DEFAULT_HTTP_TIMEOUT_SECONDS = float(os.getenv("KIOSK_CORE_HTTP_TIMEOUT_SECONDS", "300.0"))

# Wake-word activation (openwakeword)
WAKEWORD_ENABLED = os.getenv("KIOSK_CORE_WAKEWORD_ENABLED", "false").lower() not in ("false", "0", "no")
DEFAULT_WAKEWORD_MODEL = os.getenv("KIOSK_CORE_WAKEWORD_MODEL", "hey jarvis")
DEFAULT_WAKEWORD_THRESHOLD = float(os.getenv("KIOSK_CORE_WAKEWORD_THRESHOLD", "0.5"))
DEFAULT_WAKEWORD_VAD_THRESHOLD = float(os.getenv("KIOSK_CORE_WAKEWORD_VAD_THRESHOLD", "0.4"))
DEFAULT_WAKEWORD_PATIENCE_FRAMES = int(os.getenv("KIOSK_CORE_WAKEWORD_PATIENCE_FRAMES", "2"))
DEFAULT_WAKEWORD_TIMEOUT_SECONDS = float(os.getenv("KIOSK_CORE_WAKEWORD_TIMEOUT_SECONDS", "0"))
DEFAULT_WAKEWORD_INFERENCE_FRAMEWORK = os.getenv("KIOSK_CORE_WAKEWORD_INFERENCE_FRAMEWORK", "onnx")

# Speaker diarization — master switch and semantic fallback sensitivity.
# Set KIOSK_CORE_DIARIZATION_ENABLED=false to revert to flat-text behavior
# (no speaker filtering; all segments forwarded as-is).
DEFAULT_DIARIZATION_ENABLED = os.getenv("KIOSK_CORE_DIARIZATION_ENABLED", "true").lower() not in ("false", "0", "no")
# Diarization on INTERMEDIATE chunks only (the max-chunk-size-cap flush and
# the adaptive-pause pre-warm flush — i.e. every chunk except the final tail
# at true endpoint, which always keeps full diarization regardless of this
# flag). Off by default: diarization adds ~150-220ms/chunk, which pushes the
# adaptive flush's ASR round-trip past DEFAULT_ENDPOINT_SHORT_SECONDS and
# effectively disables the sentence-completeness endpoint shortcut (see
# DEFAULT_ENDPOINT_SHORT_SECONDS below) — measured landing at ~1.40-1.50s vs.
# the 1.5s silence timeout, leaving no margin for it to ever fire early.
# With diarization off on intermediate chunks only, ASR-only lands at
# ~1.23-1.28s, giving the shortcut real room to save ~200-270ms on utterances
# that read as finished.
# TRADEOFF (accepted): an intermediate chunk's transcribed text still becomes
# part of the final committed transcript (see _flush_chunk in
# audio_session.py) — it is not a throwaway heuristic check. With diarization
# off there, any bystander/secondary speech captured during that portion of
# the utterance is NOT filtered out for that slice, unlike the final tail
# chunk which always runs full diarization. Accepted as a reasonable
# trade-off since the tail (last ~0.3-0.5s before endpoint) still enforces
# speaker-filtering, and any covered utterance is short (a few seconds).
DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED = os.getenv(
    "KIOSK_CORE_DIARIZATION_INTERMEDIATE_ENABLED", "false"
).lower() not in ("false", "0", "no")

# ── Enrollment priming ──────────────────────────────────────────────────────
# The analyzer can only tag segments with is_primary once it has ENROLLED a
# reference voice for the conversation, and it enrolls from a diarized chunk
# containing a speech span of at least DEFAULT_DIARIZATION_ENROLL_MIN_SECONDS.
# The final tail chunk is usually far shorter than that ("classic chicken
# burger", ~1.2s), so if no intermediate chunk is ever diarized the analyzer
# never enrolls, is_primary is never set on any segment, and the speaker
# filter silently degrades to its first-speaker/semantic fallback — losing the
# bystander protection described under DEFAULT_SPEAKER_STRICT_DROP.
#
# This previously worked only by accident: kiosk-core's per-request
# `diarization` flag was not a parameter of the analyzer's endpoint, so it was
# discarded and EVERY chunk was diarized regardless of the flag above. With the
# flag honoured, enrollment has to be requested deliberately.
#
# So: diarize intermediate chunks until the conversation has an enrolled voice,
# then stop. The cost is paid once per conversation, on an early chunk while
# the customer is still talking — never on the final tail chunk that sits on
# the voice-to-voice critical path.
DEFAULT_DIARIZATION_ENROLLMENT_PRIMING_ENABLED = os.getenv(
    "KIOSK_CORE_DIARIZATION_ENROLLMENT_PRIMING_ENABLED", "true"
).lower() not in ("false", "0", "no")
# Must match the analyzer's own minimum enrollment span (its
# models.diarization.enrollment min duration). A shorter chunk cannot enroll,
# so diarizing it would buy nothing and only add latency.
DEFAULT_DIARIZATION_ENROLL_MIN_SECONDS = float(
    os.getenv("KIOSK_CORE_DIARIZATION_ENROLL_MIN_SECONDS", "2.0")
)

# ── Skip empty final tail-chunk ASR call ────────────────────────────────────
# Tier 2 roadmap item #4 ("cumulative-snapshot ASR"), adapted to this
# codebase's actual bottleneck rather than the reference lab's design
# (kiosk-voice-lab-main has neither per-chunk streaming ASR nor per-chunk
# diarization, so its periodic-snapshot mechanism does not transplant safely
# here — see docs/discussion for the rejected full rework).
#
# By the time the endpoint fires, the adaptive-pause flush (see
# DEFAULT_ADAPTIVE_FLUSH_PAUSE_SECONDS) has almost always already sent every
# real word to the analyzer; the final "is_final=True" tail chunk enqueued in
# _process_frame_stream typically contains nothing but the trailing silence
# frames accumulated since that flush. Whisper still pays its fixed
# per-call round-trip (~1.2-4.7s observed) to transcribe that silence into an
# empty string — pure critical-path latency with no transcript benefit.
#
# When enabled, a chunk with zero speech frames since the last flush (tracked
# per-turn in BaseAudioSession._chunk_has_speech) skips the final flush
# entirely instead of queuing it. This is safe for the speaker-filter
# invariant noted above (DEFAULT_DIARIZATION_INTERMEDIATE_ENABLED): silence
# yields no segments to filter regardless, so no verification is lost, only a
# no-op HTTP round trip. If the tail DOES contain any unflushed speech (e.g. a
# short trailing utterance under the adaptive flush's 0.5s minimum), the
# final flush still runs exactly as before — diarization coverage on real
# speech is unchanged.
#
# Default OFF pending live A/B measurement against the current behavior.
DEFAULT_SKIP_EMPTY_FINAL_FLUSH_ENABLED = os.getenv(
    "KIOSK_CORE_SKIP_EMPTY_FINAL_FLUSH_ENABLED", "false"
).lower() not in ("false", "0", "no")
# Minimum domain-keyword overlap ratio to accept a fallback segment when the
# primary customer is silent for an entire chunk.
DEFAULT_SEMANTIC_FALLBACK_THRESHOLD = float(os.getenv("KIOSK_CORE_SEMANTIC_FALLBACK_THRESHOLD", "0.10"))
# When the audio-analyzer holds an enrolled reference voice it tags every
# segment with is_primary. If it marks them all non-primary that is a positive
# rejection and the chunk is dropped, so a bystander cannot inject orders.
# Set KIOSK_CORE_SPEAKER_STRICT_DROP=false to fall back to the first-speaker /
# semantic heuristics instead — an escape hatch for when voice enrollment is
# mistuned and starts rejecting the real customer.
DEFAULT_SPEAKER_STRICT_DROP = os.getenv("KIOSK_CORE_SPEAKER_STRICT_DROP", "true").lower() not in ("false", "0", "no")

# Minimum number of repeated words required before the transcript backstop
# (BaseAudioSession._strip_duplicate_prefix) treats a leading run as an
# analyzer re-transcription rather than genuine speech. The analyzer is
# cumulative, and the flat-text and segment dedup paths track different
# cursors, so a segment straddling a previous flush can re-deliver words that
# were already committed. Three words is high enough that natural repetition
# ("yes yes", "two two please") is never swallowed.
DEFAULT_DUPLICATE_PREFIX_MIN_WORDS = int(os.getenv("KIOSK_CORE_DUPLICATE_PREFIX_MIN_WORDS", "3"))

# Spoken replies used when a turn produces no usable transcript. The two cases
# are NOT interchangeable and must never share a message:
#   * NO_SPEECH   — the microphone captured nothing (true silence). Prompting
#                   the customer to order is the right response.
#   * UNRECOGNIZED— speech WAS captured but every segment was rejected by the
#                   speaker filter (analyzer marked it non-primary, or it came
#                   from a bystander). Replying with the generic greeting here
#                   is actively misleading: the customer spoke, was ignored,
#                   and is given no hint that they need to retry.
DEFAULT_NO_SPEECH_PROMPT = os.getenv(
    "KIOSK_CORE_NO_SPEECH_PROMPT",
    "How can I help you?",
)
DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT = os.getenv(
    "KIOSK_CORE_UNRECOGNIZED_SPEAKER_PROMPT",
    "Sorry, I couldn't clearly recognise your voice. Could you please repeat that?",
)

# A single rejected turn is unreliable evidence of a real bystander — it is
# just as often a Whisper hallucination or TTS echo bleeding into the mic from
# the kiosk's own previous reply (see the rationale where this constant is
# consumed, in BaseAudioSession._finalize_run). Speaking
# DEFAULT_UNRECOGNIZED_SPEAKER_PROMPT after every single rejection reintroduces
# that false-positive noise. Requiring this many CONSECUTIVE rejected turns in
# the SAME conversation (tracked by agent_session_id, reset the moment a turn
# produces a real transcript) before speaking distinguishes a persistent
# bystander/misconfigured enrollment from a one-off echo, while still telling
# a genuinely ignored customer something after a couple of silently dropped
# turns rather than leaving them with no feedback at all.
DEFAULT_CONSECUTIVE_REJECTION_THRESHOLD = int(
    os.getenv("KIOSK_CORE_CONSECUTIVE_REJECTION_THRESHOLD", "2")
)

# ── Ordering & Agent feature ─────────────────────────────────────────────────
# Set KIOSK_CORE_ORDERING_ENABLED=false to disable the ordering/agent feature
# and keep the legacy RAG-only Q&A flow.
ORDERING_ENABLED = os.getenv("KIOSK_CORE_ORDERING_ENABLED", "true").lower() not in ("false", "0", "no")

# Single shared kiosk identity used for ordering when no per-user login is
# wired into the request (this kiosk currently serves one customer at a time).
DEFAULT_ORDERING_USER_ID = os.getenv("KIOSK_CORE_DEFAULT_USER_ID", "kiosk-user")

# RAG-service agent chat endpoint (for ordering turns).
DEFAULT_AGENT_URL = os.getenv(
    "KIOSK_CORE_AGENT_URL",
    "http://127.0.0.1:8020/api/v1/agent/chat",
)

# Consume the agent's streaming endpoint so complete sentences reach TTS as
# they are generated instead of after the whole turn.
#
# Measured: the first sentence of a reply exists at ~700 ms while the full
# reply takes 1.3-4.0 s, so time-to-first-audio drops from ~5.0 s to ~2.8 s.
#
# OFF by default, and it must stay paired with AGENT_STREAM_SENTENCES on
# rag-service: the agent only releases sentences that provably cannot be
# rewritten by a later guard, and this client still re-validates the
# authoritative reply before speaking any remainder. Set to false to fall back
# to the buffered endpoint without a rebuild.
AGENT_STREAM_ENABLED = os.getenv(
    "KIOSK_CORE_AGENT_STREAM_ENABLED", "false"
).lower() in ("true", "1", "yes")

# Derived from DEFAULT_AGENT_URL so both point at the same service.
DEFAULT_AGENT_STREAM_URL = os.getenv(
    "KIOSK_CORE_AGENT_STREAM_URL",
    DEFAULT_AGENT_URL.rstrip("/") + "/stream",
)

# SQLite database file path (ordering domain).
KIOSK_DB_PATH = os.getenv("KIOSK_CORE_DB_PATH", "./kiosk.db")

# YAML seed files for product catalogue and upsell rules.
PRODUCTS_YAML_PATH = os.getenv(
    "KIOSK_CORE_PRODUCTS_YAML",
    "./configs/ordering/products.yaml",
)
UPSELL_RULES_YAML_PATH = os.getenv(
    "KIOSK_CORE_UPSELL_RULES_YAML",
    "./configs/ordering/upsell_rules.yaml",
)
# Maximum number of upsell suggestions attached to a place_order/update_order
# result.  Every extra suggestion the agent has to speak costs ~8 output tokens
# (~350 ms of LLM decode on Panther Lake iGPU), so this directly trades upsell
# breadth against spoken-reply latency.  1 keeps the turn inside the 3-4 s SLA.
UPSELL_MAX_SUGGESTIONS = int(os.getenv("KIOSK_CORE_UPSELL_MAX_SUGGESTIONS", "1"))

# ── Identity / biometric authentication feature ──────────────────────────────
# Master switch for the multimodal (face + voice) identity subsystem.  When
# false, kiosk-core does not mount the identity router, does not construct the
# IdentityClient, and the standalone identity-service container is never called.
# Set KIOSK_CORE_IDENTITY_ENABLED=true to turn the feature on (the
# identity-service container must also be started, e.g. via the `identity`
# compose profile).
IDENTITY_ENABLED = os.getenv("KIOSK_CORE_IDENTITY_ENABLED", "false").lower() not in ("false", "0", "no")

# Base URL of the standalone identity-service.  Within Docker the service is
# reachable as http://identity-service:8013.
IDENTITY_SERVICE_URL = os.getenv(
    "KIOSK_CORE_IDENTITY_URL",
    "http://127.0.0.1:8013",
)

# ---------------------------------------------------------------------------
# Queue-service integration (dynamic peak-hour menu)
# ---------------------------------------------------------------------------
# When enabled, the queue-service exposes a queue count that kiosk-core can
# query (future server-side menu filtering).  The UI also polls this directly
# via /queue-svc/api/v1/queue/count proxied through nginx.
QUEUE_SERVICE_ENABLED = os.getenv("KIOSK_CORE_QUEUE_SERVICE_ENABLED", "true").lower() not in ("false", "0", "no")

QUEUE_SERVICE_URL = os.getenv(
    "KIOSK_CORE_QUEUE_SERVICE_URL",
    "http://127.0.0.1:8090",
)

# ---------------------------------------------------------------------------
# Conversation recording (offline analysis)
# ---------------------------------------------------------------------------
# Single master switch: when false (default), kiosk_core.conversation_recorder
# does no file I/O at all -- every call is a no-op. When true, every completed
# voice turn (user transcript + assistant reply) is appended as one JSON line
# to <CONVERSATION_LOG_DIR>/<conversation_id>.jsonl, so each full multi-turn
# conversation lives in its own file for later analysis.
CONVERSATION_LOGGING_ENABLED = os.getenv(
    "KIOSK_CORE_CONVERSATION_LOGGING_ENABLED", "false"
).lower() not in ("false", "0", "no")

# Directory conversation transcripts are written to. Relative paths resolve
# against the kiosk-core project root (same convention as KIOSK_DB_PATH).
CONVERSATION_LOG_DIR = os.getenv(
    "KIOSK_CORE_CONVERSATION_LOG_DIR",
    "./conversations",
)

# ---------------------------------------------------------------------------
# VLM metrics logging (performance-tools' vlm_metrics_logger, pip-installed)
# ---------------------------------------------------------------------------
# Emits one real start/end pair per completed voice turn -- last customer word
# to first reply audio -- via vlm_metrics_logger (installed straight from
# performance-tools' git repo -- see requirements.txt -- rather than vendored
# under kiosk_core), the same plain-text log format performance-tools'
# consolidate_multiple_run_of_metrics.py already parses. On by default: this
# is a pure side-channel append-only log write guarded by its own try/except
# (see audio_session._emit_vlm_metrics), so a failure here can never affect a
# live customer turn.
VLM_METRICS_ENABLED = os.getenv(
    "KIOSK_CORE_VLM_METRICS_ENABLED", "true"
).lower() not in ("false", "0", "no")

# Directory the logger writes vlm_application_metrics_*.txt into.
# Vlm_metrics_logger itself reads this from CONTAINER_RESULTS_PATH (an
# upstream-fixed env var name, not KIOSK_CORE_-prefixed -- it is the same
# contract performance-tools' own scripts and order-accuracy/take-away use),
# so this is set at process env level (docker-compose/.env), not read here.
# Documented alongside the other flag purely so both live in one place.
VLM_METRICS_RESULTS_DIR = os.getenv("CONTAINER_RESULTS_PATH", "./results")

# Env var NAME (not value) vlm_metrics_logger looks up via os.getenv() to
# populate the logged "application" field -- matches the label the benchmark
# harness already used (tests/benchmarks/v2v_scripted_conversation_benchmark.py
# VLM_USECASE_ENV_VAR), so turns emitted live by kiosk-core and turns
# previously synthesized by the harness land under the same "application"
# value and consolidate identically.
VLM_METRICS_USECASE_ENV_VAR = "USECASE_V2V"
