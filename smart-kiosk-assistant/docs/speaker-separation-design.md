# Speaker Separation — Design

## Problem Statement

In crowded kiosk environments multiple people may speak simultaneously — nearby customers, staff, or adjacent kiosks. Without speaker isolation, all voices collapse into a single transcript and the kiosk responds to whoever spoke loudest or last, not the person standing at the screen. This feature identifies the **primary customer** (the first coherent speaker detected in a session) and discards all other voices before the transcript reaches the AI backend.

---

## Solution Overview

Speaker Diarization (via Pyannote) labels each time segment of audio with a speaker ID (e.g. `SPEAKER_00`, `SPEAKER_01`). The Whisper ASR model produces word segments with timestamps; Pyannote independently produces speaker turn timelines. These are merged by aligning the **midpoint** of each ASR segment to the speaker turn that covers that timestamp.

The kiosk then locks onto the speaker ID of the **first chronologically meaningful utterance** in a session — the person who spoke first is the customer — and discards all subsequent segments from other speakers. This is the key design difference from the Smart Classroom application, which identifies the primary speaker retrospectively by total words spoken (teacher = most text). For a kiosk, waiting until the end of the session to identify the customer is not viable; the first speaker must be locked in immediately.

---

## Component Changes Required

### Audio Analyzer Service
The existing ASR endpoint runs Whisper for transcription and optionally Pyannote for speaker diarization, but three gaps prevent speaker labels from reaching the kiosk today:

- **Diarization is a server-side global.** It cannot be toggled per request. The endpoint must accept a `diarization` flag so the kiosk can opt in.
- **Speaker labels are stripped before the response is sent.** Even when Pyannote runs and attaches a speaker ID to each segment internally, the REST handler removes the label before serializing the JSON. This must be preserved.
- **The default response format omits segments entirely.** Only the `verbose_json` response format returns the per-segment list. The endpoint must honour this format when the kiosk requests it.
- **Config** must have diarization enabled (`diarization: true`) with a valid Hugging Face token for the Pyannote model.

### Kiosk Core
Three changes in the kiosk core consume the diarized output:

- **ASR client** — updated to request `verbose_json` and the diarization flag, and to return the full JSON payload (including segments) rather than just the transcript text.
- **Session state** — each audio session tracks a `primary_speaker_id` (initially unset) and a `pending_segments` buffer. Both persist across chunk boundaries for the lifetime of the session.
- **Speaker filter** — a new filtering step runs on every chunk between ASR and the RAG query, using the algorithm described below.

---

## Speaker Assignment Logic

The algorithm is adapted from the Smart Classroom's speaker resolution pattern (midpoint matching + pending segment buffering), simplified for the kiosk's two-class problem: primary customer vs. background noise.

### Phase 1 — Segment Matching (per chunk, within audio-analyzer)

For each ASR segment returned by Whisper, compute its **midpoint** in time and find the Pyannote speaker turn that covers that midpoint. This gives each ASR segment a speaker label. Segments that fall in a gap between turns (Pyannote detected silence or overlapping speech) are labelled `UNKNOWN`.

```
ASR segment: { text, start=1.0, end=2.4 }
                         midpoint = 1.7 s
                              │
        Pyannote turns:  ─────┼────────────────────────────────
                         [0.5─2.0: SPEAKER_00] [2.5─4.0: SPEAKER_01]
                              │
                        SPEAKER_00 covers 1.7 s → assign SPEAKER_00
```

### Phase 2 — Primary Speaker Lock-On (within kiosk core)

```
Segment arrives from audio-analyzer
        │
        ▼
Speaker label = SPEAKER_00 / SPEAKER_01 / UNKNOWN
        │
        │  UNKNOWN means Pyannote found no speaker turn covering this
        │  segment's midpoint — it landed in a silence gap or overlapping
        │  speech window where Pyannote could not assign a single speaker.
        │
        ├─ UNKNOWN
        │       │
        │       ▼
        │   Has primary_speaker_id been set yet?
        │       │
        │       ├─ NO  (UNKNOWN arrived before any known speaker)
        │       │       We have no idea who this is.
        │       │       Buffer into pending_segments.
        │       │       When the first known speaker is locked in later,
        │       │       all pending segments get retroactively assigned
        │       │       to that primary (they were likely the same person
        │       │       speaking before Pyannote could resolve the turn).
        │       │
        │       └─ YES (UNKNOWN arrived after primary is already locked)
        │               We treat it as a continuation of the primary.
        │               Pyannote simply couldn't label this moment cleanly,
        │               but the last known voice was the primary customer
        │               so we give them the benefit of the doubt → KEEP
        │
        └─ KNOWN speaker (SPEAKER_00 / SPEAKER_01 / …)
                │
                ▼
        Is primary_speaker_id set?
                │
         NO ────┴──► Is this segment meaningful?
                     (more than 1 non-punctuation character)
                            │
                      YES ──┴──► Lock: primary_speaker_id = this speaker
                                 Flush pending_segments:
                                   assign them all to primary_speaker_id
                                   (retroactive assignment — same as
                                    Smart Classroom pending flush)
                                 KEEP this segment
                      NO  ──────► Discard (Whisper noise token)
                │
         YES ───┴──► speaker == primary_speaker_id?
                            │
                      YES ──┴──► KEEP segment
                      NO  ──────► DISCARD (background voice)
```

### Phase 3 — Domain-Keyword Fallback

If an entire chunk produces zero kept segments (the primary customer was silent but background voices were active), a lightweight keyword-overlap check is applied to each discarded segment. If a segment's text overlaps sufficiently with kiosk domain vocabulary (menu items, payment, ordering terms), it is accepted and `primary_speaker_id` is reassigned to that segment's speaker. This handles the edge case where the original customer steps away and a new customer begins at the kiosk.

---

## Worked Example — Primary + Secondary Speaker in the Same Chunk

**Scenario:** Primary customer says *"I want to order pizza"* and a nearby person whispers *"order cold drink too"* at nearly the same time.

**What Pyannote + Whisper produce:**

```
Pyannote speaker turns:
  [0.0 – 2.1 s]  SPEAKER_00   ← primary customer
  [2.3 – 3.6 s]  SPEAKER_01   ← background person

Whisper ASR segments:
  { text: "I want to order pizza",  start: 0.2, end: 2.0 }   midpoint = 1.1 s
  { text: "order cold drink too",   start: 2.4, end: 3.5 }   midpoint = 2.95 s
```

**After midpoint matching (inside audio-analyzer):**

```
"I want to order pizza"   → midpoint 1.1 s → SPEAKER_00 turn [0.0–2.1] → speaker: SPEAKER_00
"order cold drink too"    → midpoint 2.95 s → SPEAKER_01 turn [2.3–3.6] → speaker: SPEAKER_01
```

**Speaker filter decision (inside kiosk core):**

```
primary_speaker_id = SPEAKER_00  (already locked from earlier)

Segment 1: speaker=SPEAKER_00 == primary → ✓ KEEP
Segment 2: speaker=SPEAKER_01 != primary → ✗ DISCARD

kept_segments is NOT empty → Phase 3 fallback does NOT run
(even though "order cold drink too" contains domain keywords,
 the fallback is only invoked when the primary spoke zero words
 in the entire chunk)
```

**Result sent to RAG:**

```
filtered_text = "I want to order pizza"
```

The secondary speaker's request is silently dropped. The kiosk orders only the pizza.

**Why the fallback does not save the secondary speaker here:**  
The domain-keyword fallback exists for the case where the *primary* customer is completely silent in a chunk and only background voices are heard. When the primary and secondary both speak in the same chunk, the primary's segments are kept and the fallback is not evaluated at all — there is no ambiguity to resolve.

---

## Comparison with Smart Classroom

| Aspect | Smart Classroom | Kiosk |
|--------|-----------------|-------|
| Primary speaker identified by | Most total words spoken (retrospective, end of session) | First chronological meaningful utterance (immediate) |
| Unknown-speaker segments | Buffered, then assigned to `last_known_speaker` | Buffered pre-lock; post-lock inherit `primary_speaker_id` |
| Non-primary speakers | Labelled `STUDENT_0`, `STUDENT_1` and kept | Discarded entirely |
| Output | Full multi-speaker labelled transcript | Customer-only text string |
| Session scope | Full lecture (minutes to hours) | Single interaction (seconds to ~20 s) |

---

## Sequence Flow

> **Transport summary:**
> - **Gradio UI → Kiosk Core (audio):** HTTP POST chunks, no streaming
> - **Gradio UI → Kiosk Core (status):** HTTP polling every 350 ms — no SSE, no websockets
> - **Kiosk Core → Audio Analyzer:** HTTP POST, synchronous request/response
> - **Kiosk Core → RAG Service:** genuine SSE (`Accept: text/event-stream`, `data:` prefixed lines) — tokens are streamed back and accumulated as they arrive, confirmed in `rag_client.py`
> - **Kiosk Core → TTS Service:** HTTP POST, returns WAV file

```
[Gradio UI]                [Kiosk Core]           [Audio Analyzer]   [RAG Service]   [TTS Service]
     │                          │                        │                  │               │
     │  POST /start-stream      │                        │                  │               │
     │─────────────────────────►│                        │                  │               │
     │  ◄── { session_id }  ────│                        │                  │               │
     │                          │                        │                  │               │
     │  POST /audio (chunk 1)   │                        │                  │               │
     │─────────────────────────►│                        │                  │               │
     │  POST /audio (chunk 2)   │  POST /transcriptions  │                  │               │
     │─────────────────────────►│ ──────────────────────►│                  │               │
     │        …                 │  diarization=true      │ Whisper ASR      │               │
     │  POST /audio/end         │  response_format=      │ Pyannote Diarize │               │
     │─────────────────────────►│    verbose_json        │ Midpoint match   │               │
     │                          │ ◄── segments [{        │                  │               │
     │                          │   speaker, text,       │                  │               │
     │                          │   start, end }]        │                  │               │
     │                          │                        │                  │               │
     │                          │  _filter_target_speaker()                 │               │
     │                          │  → lock/filter by primary_speaker_id      │               │
     │                          │  → filtered_text                          │               │
     │                          │                                           │               │
     │                          │  POST /api/v1/query ──────────────────────►               │
     │                          │  { transcription, history }               │               │
     │                          │  ◄── SSE token stream ─────────────────────               │
     │                          │  (tokens accumulated into response)       │               │
     │                          │                                           │               │
     │                          │  POST /v1/audio/speech ────────────────────────────────── ►│
     │                          │  ◄── WAV file ─────────────────────────────────────────── ─│
     │                          │  (saved to generated_audio/{session_id}/) │               │
     │                          │                                           │               │
     │  GET /sessions/{id}      │                                           │               │
     │─────────────────────────►│  ◄── { status, transcript,                │               │
     │  (poll every 350 ms)     │        response, tts_audio_segments }     │               │
     │  ◄── snapshot ───────────│                                           │               │
     │  … repeat until          │                                           │               │
     │    status = completed ───│                                           │               │
     │                          │                                           │               │
     │  render chat + play audio│                                           │               │
```

---

## Graceful Degradation

| Condition | Behaviour |
|-----------|-----------|
| Audio analyzer diarization disabled | No speaker labels in segments; all segments treated as primary; feature is bypassed transparently |
| Single speaker in audio | Primary locked on first segment; all subsequent chunks pass through with no filtering overhead |
| Primary speaker quiet in a chunk | Domain-keyword fallback decides whether to accept another speaker or drop the chunk |
| Audio analyzer returns no segments | Falls back to flat transcript text; no filtering applied |
| Whisper noise tokens (`[BLANK_AUDIO]`, `[Music]`, etc.) | Stripped by existing noise filter after speaker filtering |

---

## Deployment Dependencies

| Dependency | Requirement |
|------------|-------------|
| Pyannote speaker diarization model | Must be downloaded and accessible to the audio analyzer container |
| Hugging Face token | Required to download the Pyannote model; set in audio analyzer config |
| Audio analyzer config | `models.asr.diarization: true` |
| Network | Kiosk core → audio analyzer latency should remain under the configured ASR timeout (default 120 s) |
