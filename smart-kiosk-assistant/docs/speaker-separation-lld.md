# Low-Level Design: Speaker Separation in Smart Kiosk Assistant

## 1. Overview

In crowded kiosk environments multiple people may speak simultaneously — nearby customers, staff, or adjacent kiosks. This feature identifies and locks onto the **primary customer** (the first coherent speaker detected in a session) and discards segments from all other voices before forwarding the transcript to the RAG service.

---

## 2. System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                       Smart Kiosk Assistant                     │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              BaseAudioSession  (audio_session.py)        │   │
│  │                                                          │   │
│  │  _process_frame_stream()                                 │   │
│  │       │                                                  │   │
│  │       └──► _flush_chunk()                                │   │
│  │                 │                                        │   │
│  │                 ├── AnalyzerClient.transcribe_file()  ───┼──►│  audio-analyzer :8010
│  │                 │        (diarization=True)              │   │  POST /v1/audio/transcriptions
│  │                 │                                        │   │  form: diarization="true"
│  │                 ├── _filter_target_speaker(segments)     │   │
│  │                 │        ├── assign primary_speaker_id   │   │
│  │                 │        ├── filter by speaker ID        │   │
│  │                 │        └── semantic fallback           │   │
│  │                 │                                        │   │
│  │                 └── transcript_parts.append(text)        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            │                                    │
│                            ▼                                    │
│               RagClient.stream_answer()  ──────────────────────►│  rag-service :8020
│                            │                                    │
│                            ▼                                    │
│               TtsClient.synthesize_to_file()  ─────────────────►│  tts-service :8011
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Audio-Analyzer Service Interface

> **IMPORTANT — current limitations of the existing endpoint.** The existing
> `POST /v1/audio/transcriptions` endpoint does **not** support per-request
> speaker separation today. Three gaps must be closed in the audio-analyzer
> before the kiosk-side logic can work (see [§9](#9-audio-analyzer-service--required-changes)):
>
> 1. The endpoint signature accepts only `file, model, session_id, language,
>    prompt, response_format, temperature`. There is **no `diarization` form
>    field** — FastAPI silently drops unknown fields, so it has zero effect.
>    Diarization is currently a server-side global (`config.models.asr.diarization`)
>    evaluated once at import time.
> 2. With the default `response_format="json"`, the endpoint returns **only**
>    `{"text": ...}` — no `segments`. Segments are returned exclusively for
>    `response_format="verbose_json"`.
> 3. The REST path calls `Pipeline.transcribe()`, which builds verbose
>    segments with `include_speaker=False`, so even `verbose_json` strips the
>    `"speaker"` key. Only the unused streaming path passes `include_speaker=True`.

### 3.1 Request (target design)

| Field             | Type                  | Value            | Notes                                                      |
|-------------------|-----------------------|------------------|------------------------------------------------------------|
| `POST`            | URL                   | `http://audio-analyzer:8010/v1/audio/transcriptions` | Configured via `KIOSK_CORE_ANALYZER_URL` |
| Content-Type      | `multipart/form-data` | —                |                                                            |
| `file`            | binary                | WAV audio chunk  | 16-bit PCM, mono, 16 kHz                                   |
| `temperature`     | string                | `"0.0"`          | Whisper decoding temperature                               |
| `language`        | string (optional)     | e.g. `"en"`      | Passed through from session request                        |
| `response_format` | string                | `"verbose_json"` | **Required** — only this format returns the `segments` list |
| `diarization`     | string                | `"true"`         | **New field** — must be added to the endpoint (see §9)     |

### 3.2 Response Schema (`verbose_json`, diarized)

The real `verbose_json` segment carries extra Whisper fields. The kiosk only
reads `speaker` and `text`; the remaining keys are ignored but documented here
for accuracy.

```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 5.1,
  "text": "Hello, I would like to order a cheeseburger. Do you want fries with that?",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.2,
      "text": "Hello, I would like to order a cheeseburger.",
      "tokens": [],
      "temperature": 0.0,
      "avg_logprob": -0.21,
      "compression_ratio": 1.4,
      "no_speech_prob": 0.01,
      "speaker": "SPEAKER_00"
    },
    {
      "id": 1,
      "seek": 0,
      "start": 3.5,
      "end": 5.1,
      "text": "Do you want fries with that?",
      "tokens": [],
      "temperature": 0.0,
      "avg_logprob": -0.30,
      "compression_ratio": 1.1,
      "no_speech_prob": 0.02,
      "speaker": "SPEAKER_01"
    }
  ]
}
```

> **Backward compatibility:** When diarization is disabled on the audio-analyzer
> side, segments will not contain a `"speaker"` key. The filtering logic degrades
> gracefully — it treats every segment as belonging to the primary speaker.

---

## 4. Data Flow — Per Audio Chunk

```
Audio frames (numpy int16)
        │
        ▼
_flush_chunk(frames)
        │
        ├─ 1. Concatenate frames → write temp WAV
        │
        ├─ 2. POST to audio-analyzer
        │       form: diarization=true, response_format=verbose_json
        │       Returns { "text": "...", "segments": [ {speaker, text, ...} ] }
        │
        ├─ 3. _filter_target_speaker(segments)
        │       ├─ [First chunk] primary_speaker_id = None
        │       │       → pick speaker of first non-noise segment
        │       │       → set self.primary_speaker_id = "SPEAKER_00"
        │       │
        │       ├─ [Subsequent chunks] filter segments where
        │       │       segment["speaker"] == primary_speaker_id
        │       │
        │       ├─ [Ambiguous / missing speaker] semantic fallback
        │       │       → keyword similarity check against history
        │       │
        │       └─ Concatenate kept segment texts → return final_text
        │
        ├─ 4. Strip Whisper hallucination tokens (_WHISPER_JUNK regex)
        │
        └─ 5. Append final_text to self.transcript_parts
```

---

## 5. Modified Files & Classes

### 5.1 `kiosk_core/analyzer_client.py` — `AnalyzerClient`

#### Changed method: `transcribe_file`

| Aspect          | Before                                    | After                                                                       |
|-----------------|-------------------------------------------|------------------------------------------------------------------------------|
| Return type     | `str`                                     | `dict` (raw JSON payload)                                                   |
| New parameter   | —                                         | `diarization: bool = False`                                                |
| Form field sent | `temperature`, `language`                 | `temperature`, `language`, `response_format="verbose_json"`, `diarization` |
| Extraction      | `payload.get("text", "")` and returns str | Returns full `response.json()` dict                       |

```python
def transcribe_file(
    self,
    file_path: str,
    language: str | None = None,
    temperature: float = 0.0,
    diarization: bool = False,          # NEW
) -> dict:                              # Return type changed: str → dict
    ...
    data = {"temperature": str(temperature)}
    if language:
        data["language"] = language
    if diarization:
        data["diarization"] = "true"
        data["response_format"] = "verbose_json"   # REQUIRED to receive segments
    ...
    return response.json()              # Full JSON dict, not just "text"
```

---

### 5.2 `kiosk_core/audio_session.py` — `BaseAudioSession`

#### New attribute

```python
self.primary_speaker_id: str | None = None
self.pending_segments: list[dict] = []
```

Added in `__init__`. Both attributes persist across chunk flushes for the lifetime of the session.
- `primary_speaker_id = None` means the first meaningful speaker has not been identified yet.
- `pending_segments` buffers segments whose speaker was `UNKNOWN` (fell in a Pyannote gap) before the primary is established. This pattern is taken directly from the Smart Classroom ASR component's `pending_segments` + `last_known_speaker` buffering.

#### New method: `_filter_target_speaker`

```python
def _filter_target_speaker(self, segments: list[dict]) -> str:
```

**Inputs:** `segments` — list of diarized segment dicts from the ASR response, each optionally containing `"speaker"`, `"text"`, `"start"`, `"end"`. The `"speaker"` key is either a Pyannote speaker label (`SPEAKER_00`, `SPEAKER_01`, …) or absent/`None` when the ASR segment midpoint fell in a gap between Pyannote speaker turns (i.e. unresolvable).

**Outputs:** Concatenated text of the segments that belong to the primary speaker.

**Algorithm:**

```
_filter_target_speaker(segments)
        │
        ├─ [Guard] If segments is empty → return ""
        │
        ├─ For each segment in chronological order:
        │
        │   speaker = segment.get("speaker")   # None = UNKNOWN turn gap
        │
        │   ── Case A: UNKNOWN speaker ─────────────────────────────────────
        │   │
        │   │  primary_speaker_id already set?
        │   │       YES → assign segment to primary_speaker_id (inherit,
        │   │              same as Smart Classroom last_known_speaker logic)
        │   │              → KEEP
        │   │       NO  → buffer into self.pending_segments
        │   │              (retroactive assignment when primary is locked)
        │   │
        │   ── Case B: KNOWN speaker ────────────────────────────────────────
        │   │
        │   │  primary_speaker_id not yet set?
        │   │       ├─ Is segment meaningful?
        │   │       │  (meaningful_char_count > 1, no Whisper noise token)
        │   │       │       YES → Lock: primary_speaker_id = speaker
        │   │       │              Flush pending_segments:
        │   │       │                assign all pending to primary_speaker_id
        │   │       │                move all pending to kept_segments
        │   │       │              KEEP this segment
        │   │       │       NO  → Discard (noise/punctuation only)
        │   │
        │   │  primary_speaker_id already set?
        │   │       ├─ speaker == primary_speaker_id → KEEP
        │   │       └─ speaker != primary_speaker_id → DISCARD (background voice)
        │
        ├─ [Step 3 — Semantic Fallback]
        │     If kept_segments is EMPTY after processing all segments:
        │       (primary was silent this whole chunk; only background spoke)
        │       Score each discarded segment for domain relevance:
        │
        │         score(text) = |words(text) ∩ DOMAIN_KEYWORDS|
        │                       / max(len(words(text)), 1)
        │
        │       DOMAIN_KEYWORDS: kiosk/QSR/retail vocabulary
        │         {"order", "burger", "pizza", "menu", "combo", "price",
        │          "pay", "card", "flight", "ticket", "seat", "checkout", ...}
        │
        │       If best_score > SEMANTIC_FALLBACK_THRESHOLD (0.10):
        │         Accept that segment
        │         Reassign self.primary_speaker_id = that segment's speaker
        │         (customer may have changed — new person stepped up)
        │       Else:
        │         Return "" (silent chunk, primary said nothing)
        │
        └─ Return " ".join(seg["text"] for seg in kept_segments).strip()
```

**Key design difference from Smart Classroom:** The classroom identifies the teacher *retrospectively* at the end of the session as the speaker with the most total words. The kiosk cannot wait — it locks the primary speaker on the **first chronologically meaningful utterance** so that filtering begins immediately on the second segment onwards.

**Meaningful char count** (adopted from Smart Classroom `_meaningful_char_count`):  
Counts characters that are not whitespace and not Unicode punctuation (category starting with `"P"`). A segment must have `> 1` such characters to qualify as meaningful and trigger primary speaker lock-on.

#### Modified method: `_flush_chunk`

```python
def _flush_chunk(self, frames: list[np.ndarray]) -> None:
    audio = np.concatenate(frames, axis=0)
    temp_path = self._write_temp_wav(audio)
    try:
        payload = self.client.transcribe_file(          # now returns dict
            temp_path,
            language=self.request.language,
            temperature=self.request.temperature,
            diarization=True,                           # NEW
        )
        segments: list[dict] = payload.get("segments", [])
        if segments:
            text = self._filter_target_speaker(segments)
        else:
            # audio-analyzer returned no segments — fall back to flat text
            text = str(payload.get("text", "")).strip()

        if text:
            text = _WHISPER_JUNK.sub("", text).strip()
        if text:
            with self._lock:
                self.transcript_parts.append(text)
    finally:
        Path(temp_path).unlink(missing_ok=True)
```

---

## 6. State Machine — `primary_speaker_id` + `pending_segments`

The two attributes evolve together across chunk boundaries, mirroring the Smart Classroom's `last_known_speaker` / `pending_segments` buffering pattern.

```
Session Created
      │
      ▼
primary_speaker_id = None
pending_segments   = []
      │
      │  Segment arrives with UNKNOWN speaker
      ▼
pending_segments.append(segment)   (buffer — no primary yet)
      │
      │  First meaningful KNOWN-speaker segment arrives
      ▼
primary_speaker_id = "SPEAKER_00"
Flush pending_segments → assign all to SPEAKER_00 → keep them
pending_segments = []
      │
      │  All subsequent chunks
      ▼
KNOWN speaker == SPEAKER_00 ──► KEEP
KNOWN speaker != SPEAKER_00 ──► DISCARD
UNKNOWN speaker             ──► assign to SPEAKER_00 (inherit), KEEP
      │
      │  Chunk returns 0 kept segments + semantic fallback wins
      ▼
primary_speaker_id = "SPEAKER_01"   (customer changed)
      │
Session End → both attributes discarded (not persisted)
```

> **Note:** Both `primary_speaker_id` and `pending_segments` are **in-memory only**, scoped to a single `BaseAudioSession` instance. A new kiosk interaction starts fresh every time.

---

## 7. Session Snapshot Changes

`BaseAudioSession.snapshot()` exposes the current primary speaker ID for observability:

```python
"primary_speaker_id": self.primary_speaker_id,   # str | None
```

This allows the API consumer / debug UI to see which speaker the kiosk has locked onto.

---

## 8. Configuration

All new tunables are read from environment variables with safe defaults:

| Env Variable                             | Default | Purpose                                               |
|------------------------------------------|---------|-------------------------------------------------------|
| `KIOSK_CORE_DIARIZATION_ENABLED`         | `true`  | Master switch. Set `false` to revert to flat text.    |
| `KIOSK_CORE_SEMANTIC_FALLBACK_THRESHOLD` | `0.10`  | Min domain-keyword ratio to accept a fallback segment |

When `KIOSK_CORE_DIARIZATION_ENABLED=false`, `_flush_chunk` calls `transcribe_file(diarization=False)` and skips `_filter_target_speaker` entirely, reverting to pre-feature behavior.

---

## 9. Audio-Analyzer Service — Required Changes

The audio-analyzer has the diarization *engine* (Pyannote) but the REST endpoint
does **not** expose speaker labels today. Three code changes plus one config
change are required. Without them, the kiosk receives no `segments`/`speaker`
data and the feature is a silent no-op.

### 9.1 Config (`audio-analyzer/config.yaml`)

```yaml
models:
  asr:
    diarization: true          # enable Pyannote diarization globally
    hf_token: "<YOUR_TOKEN>"   # HF token for pyannote/speaker-diarization
```

> `diarization` is read once at import time as the module global
> `ENABLE_DIARIZATION` in `components/asr_component.py`. Enabling it here turns on
> per-chunk `PyannoteDiarizer.diarize()`, which attaches a `"speaker"` key to each
> raw segment.

### 9.2 Code change 1 — accept a `diarization` form field (`api/openai_endpoints.py`)

The current endpoint signature has no `diarization` parameter, so the field is
silently dropped. Add it:

```python
@router.post("/v1/audio/transcriptions")
def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    session_id: str | None = Form(None),
    language: str | None = Form("en"),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    diarization: bool = Form(False),          # NEW
):
```

Pass the flag down to the pipeline so it can decide whether to emit speaker
labels (see 9.3). If you prefer to keep diarization purely server-config-driven,
you may skip this field — but then the kiosk cannot toggle it per request, and
`config.models.asr.diarization` alone governs behavior.

### 9.3 Code change 2 — emit speaker labels in `verbose_json` (`pipeline.py`)

`Pipeline.transcribe()` (the method the REST endpoint calls) currently builds
segments with `include_speaker=False`, which strips the `"speaker"` key even when
diarization ran:

```python
# BEFORE
verbose_segments.append(self._build_verbose_segment(index, segment, include_speaker=False))

# AFTER — preserve speaker when diarization is enabled
include_speaker = bool(getattr(config.models.asr, "diarization", False))
verbose_segments.append(
    self._build_verbose_segment(index, segment, include_speaker=include_speaker)
)
```

`_build_verbose_segment` already copies the `"speaker"` key when
`include_speaker=True` and the raw segment carries it — no change needed there.

### 9.4 Code change 3 — kiosk must request `verbose_json`

The endpoint only returns a `segments` list for `response_format="verbose_json"`.
The default `"json"` returns `{"text": ...}` only. This is handled on the kiosk
side: `AnalyzerClient.transcribe_file` sends `response_format="verbose_json"`
whenever `diarization=True` (see §5.1). No further audio-analyzer change is
needed for this — `"verbose_json"` is already in `SUPPORTED_RESPONSE_FORMATS`.

### 9.5 Graceful degradation

If `diarization: false` (or the endpoint field is omitted), segments arrive
without a `"speaker"` key. The kiosk-side `_filter_target_speaker` treats missing
labels as "keep all", so transcription still works — just without speaker
separation.

---

## 10. Sequence Diagram — Full Chunk Lifecycle

```
Kiosk Core                   audio-analyzer              rag-service
    │                              │                          │
    │  [mic/browser audio frames]  │                          │
    │  _flush_chunk()              │                          │
    │──POST /v1/audio/transcriptions ──────────────────────►  │
    │  multipart: file=chunk.wav   │                          │
    │             diarization=true │                          │
    │             response_format= │                          │
    │               verbose_json   │                          │
    │             temperature=0.0  │                          │
    │                              │                          │
    │                              │  Whisper ASR             │
    │                              │  Pyannote Diarize        │
    │◄── 200 JSON ─────────────────│                          │
    │  {text, segments:[           │                          │
    │    {speaker, text,           │                          │
    │     start, end}, ...]}       │                          │
    │                              │                          │
    │  _filter_target_speaker()    │                          │
    │  → locked to SPEAKER_00      │                          │
    │  → filtered_text             │                          │
    │                              │                          │
    │  transcript_parts.append()   │                          │
    │                              │                          │
    │  [on silence_timeout]        │                          │
    │  _stream_rag_response()      │                          │
    │──POST /api/v1/query ─────────┼─────────────────────────►│
    │  {transcription, history}    │                          │
    │◄── SSE token stream ─────────┼──────────────────────────│
    │                              │                          │
    │  TTS → speaker playback      │                          │
```

---

## 11. Files Changed Summary

### Kiosk Core (`smart-kiosk-assistant`)

| File                                 | Change Type | Description                                                                 |
|--------------------------------------|-------------|-----------------------------------------------------------------------------|
| `kiosk_core/analyzer_client.py`      | Modified    | `transcribe_file` accepts `diarization: bool`, sends `response_format="verbose_json"`, returns `dict` not `str` |
| `kiosk_core/audio_session.py`        | Modified    | Add `primary_speaker_id` attribute, `_filter_target_speaker()` method, update `_flush_chunk()` |
| `kiosk_core/config.py`               | Modified    | Add `DEFAULT_DIARIZATION_ENABLED` and `DEFAULT_SEMANTIC_FALLBACK_THRESHOLD` |

No changes are required to `models.py`, `service.py`, `rag_client.py`, or `tts_client.py`.

### Audio-Analyzer (`microservices/audio-analyzer`)

| File                          | Change Type | Description                                                                       |
|-------------------------------|-------------|----------------------------------------------------------------------------------|
| `config.yaml`                 | Modified    | Set `models.asr.diarization: true` and a valid `models.asr.hf_token`             |
| `api/openai_endpoints.py`     | Modified    | Add `diarization: bool = Form(False)` to `transcribe_audio` (optional — see §9.2) |
| `pipeline.py`                 | Modified    | In `transcribe()`, build verbose segments with `include_speaker=True` when diarization is enabled |

> Without the audio-analyzer changes (§9), segments are returned without
> `"speaker"` labels and the kiosk silently falls back to flat-text behavior.

---

## 12. Edge Cases & Guardrails

| Scenario                                      | Behaviour                                                                          |
|-----------------------------------------------|------------------------------------------------------------------------------------|
| Audio-analyzer returns no segments            | Fall back to flat `"text"` field; no filtering applied                             |
| All segments belong to non-primary speakers   | Semantic fallback kicks in; if no domain match, chunk is silently dropped          |
| Audio-analyzer diarization disabled           | `"speaker"` absent from segments; all segments treated as primary                  |
| Primary speaker stops mid-session             | Semantic fallback may reassign `primary_speaker_id` to next relevant speaker       |
| Single speaker in audio                       | `primary_speaker_id` set on first segment; all subsequent chunks pass through      |
| Whisper noise tokens (`[BLANK_AUDIO]`, etc.)  | Stripped by existing `_WHISPER_JUNK` regex after filtering, before appending       |
