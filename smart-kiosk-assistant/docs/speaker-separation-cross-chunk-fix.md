# Design: Cross-Chunk Speaker Identity Fix (Embedding-Based Diarization)

> **Status:** Proposed — not yet implemented. Captured for future work.
> **Author context:** Follow-up to `speaker-separation-lld.md` and
> `speaker-separation-design.md`. Addresses why speaker filtering never drops
> secondary-speaker text in practice.

---

## 1. Problem Statement

With diarization enabled, the kiosk is expected to lock onto the **primary
customer** and discard all other voices before forwarding the transcript to the
RAG service. In practice it **never drops any text** — every utterance is
attributed to the primary speaker, even when a second person speaks.

---

## 2. Evidence (Observed Logs)

Audio-analyzer (`asr_component.py`) — every chunk:

```
[DIARIZATION] session=... chunk=... | pyannote detected 1 speaker turn(s): SPEAKER_00[0.99s–2.95s]
[DIARIZATION] session=... | whisper produced 1 segment(s)
[DIARIZATION] segment [0.00s–3.00s] midpoint=1.50s → speaker=SPEAKER_00 | text='...'
```

Kiosk-core (`audio_session.py`) — every chunk:

```
[SPEAKER-FILTER] processing 1 segment(s), primary_speaker=NOT_SET
[SPEAKER-FILTER] seg[0] speaker=SPEAKER_00 → PRIMARY LOCKED ✓
[SPEAKER-FILTER] RESULT: kept=1 dropped=0 primary=SPEAKER_00
```

**Pattern:** every chunk yields exactly one speaker turn, always labeled
`SPEAKER_00`, which always equals the locked primary → `dropped=0` forever.

---

## 3. Root Cause

The kiosk streams audio as **independent ~5-second WAV chunks**, each sent as a
separate `POST /v1/audio/transcriptions` request. On the audio-analyzer side
each request creates a fresh `Pipeline` and runs pyannote **per file**.

pyannote assigns speaker labels (`SPEAKER_00`, `SPEAKER_01`, …) **fresh for each
audio file, with no identity preserved across files**. The dominant speaker in
*any* chunk is always labeled `SPEAKER_00`.

The kiosk's `_filter_target_speaker()` compares **label strings** across chunks:

```
Chunk 1: customer  → pyannote labels SPEAKER_00 → kiosk locks primary = "SPEAKER_00"
Chunk 2: bystander → pyannote labels SPEAKER_00 (fresh file) → "SPEAKER_00" == primary → KEEP  ✗
```

Because every chunk's dominant speaker collapses to `SPEAKER_00`, the label-string
comparison always matches the locked primary and **nothing is ever discarded**.

Label-based separation only works when two speakers appear **within the same
chunk** and pyannote splits them into `SPEAKER_00` + `SPEAKER_01`. With short
chunks and sequential (non-overlapping) speech, pyannote reports a single turn
per chunk — confirmed by the logs.

---

## 4. Why smart-classroom Doesn't Hit This (Delta Analysis)

This service's diarization was ported from
`edge-ai-suites/education-ai-suite/smart-classroom`. A file-by-file diff:

- `components/asr/diarization/pyannote_diarizer.py` — **functionally identical**
  (only type hints / comments added).
- `models.diarization` config — **identical** model
  (`pyannote/speaker-diarization-community-1`).
- `components/asr_component.py` — **this is where the real delta lives.**

| smart-classroom (source)                                   | audio-analyzer + kiosk (current)                     |
|------------------------------------------------------------|------------------------------------------------------|
| Processes the **whole recording** in one `process()` call  | Independent ~5s chunks via separate HTTP requests    |
| Maintains `last_known_speaker` / `pending_segments` across chunks within the session | Stateless per chunk; raw pyannote labels passed through |
| At the **end**, computes `teacher_speaker = max(speaker_text_len)` and relabels everyone → TEACHER / STUDENT_xx (**global batch aggregation**) | Kiosk does **real-time per-chunk label-string locking** |

Smart-classroom never performs real-time per-chunk primary locking. It tolerates
per-chunk label collisions because it aggregates over the entire recording
afterward (whoever speaks the most overall becomes the teacher). That strategy:

1. Requires the **full recording up front** — incompatible with the kiosk's
   real-time streaming, turn-by-turn RAG flow.
2. Was **not** carried over; the kiosk substituted a label-equality scheme that
   is fundamentally broken across independently-diarized chunks.

---

## 5. Proposed Solution — Speaker Embeddings (Voice Fingerprints)

A speaker **embedding** is a numerical voice fingerprint that represents *a
voice*, independent of which chunk it came from. Two embeddings from the same
person score high cosine similarity even across separate files; labels do not.

### Feasibility (verified)

`pyannote.audio == 4.0.4` (per `requirements.txt`). The
`SpeakerDiarization.apply()` path **always populates** embeddings:

```python
# pyannote/audio/pipelines/speaker_diarization.py
@dataclass
class DiarizeOutput:
    speaker_diarization: Annotation
    exclusive_speaker_diarization: Annotation
    # one speaker embedding per speaker, as (num_speakers, dimension) array,
    # sorted in speaker_diarization.labels() order
    speaker_embeddings: np.ndarray | None = None
```

`output.speaker_embeddings` returns cluster centroids — one row per speaker,
aligned with `speaker_diarization.labels()` order. No extra flag required.

---

## 6. Implementation Plan

### Phase 1 — audio-analyzer (`microservices/audio-analyzer`)

1. **`components/asr/diarization/pyannote_diarizer.py`**
   - Capture `output.speaker_embeddings`.
   - Build a `label → embedding` map using
     `output.speaker_diarization.labels()` order.
   - Include the matching embedding (L2-normalized `float32` list) on each
     returned turn dict: `{"start", "end", "speaker", "embedding"}`.

2. **`components/asr_component.py`**
   - In `process()`, when assigning a segment's speaker via midpoint matching,
     also attach the speaker's `speaker_embedding` to the emitted segment.

3. **`pipeline.py` — `_build_verbose_segment()`**
   - Pass the `speaker_embedding` key through into the verbose segment so it
     survives into the `verbose_json` REST response.

### Phase 2 — kiosk-core (`smart-kiosk-assistant/kiosk_core`)

4. **`audio_session.py`**
   - Add `self.primary_speaker_embedding: np.ndarray | None = None`.
   - On **PRIMARY LOCK**, store the locked speaker's embedding.
   - For subsequent chunks, in `_filter_target_speaker()`:
     - Compute **cosine similarity** between each segment's embedding and
       `primary_speaker_embedding`.
     - `sim >= SIMILARITY_THRESHOLD` (default **0.70**, env-configurable) → KEEP.
     - `sim <  SIMILARITY_THRESHOLD` → DISCARD (secondary voice).
   - **Graceful fallback:** if a segment carries no embedding (diarization
     disabled, older audio-analyzer, etc.), fall back to the existing
     label-string comparison so behavior never regresses.

5. **`config.py`**
   - Add `DEFAULT_SPEAKER_SIMILARITY_THRESHOLD`
     (env `KIOSK_CORE_SPEAKER_SIMILARITY_THRESHOLD`, default `0.70`).

---

## 7. Impact Summary

| File | Service | Change |
|------|---------|--------|
| `components/asr/diarization/pyannote_diarizer.py` | audio-analyzer | Return per-turn speaker embedding |
| `components/asr_component.py` | audio-analyzer | Attach embedding to each segment |
| `pipeline.py` | audio-analyzer | Pass embedding through verbose segment |
| `kiosk_core/audio_session.py` | kiosk-core | Cosine-similarity cross-chunk match + fallback |
| `kiosk_core/config.py` | kiosk-core | Similarity threshold constant |

**Rebuild required:** `intel/audio-analyzer` and `intel/kiosk-core` images.

---

## 8. Risks & Considerations

- **Short-chunk centroids:** embeddings derived from ~5s (sometimes <3s) audio
  are noisier than full-utterance embeddings. The 0.70 threshold will likely
  need empirical tuning against real two-speaker recordings.
- **Threshold tuning:** too high → primary's own text dropped on off-mic/noisy
  chunks; too low → bystanders leak through. Make it env-configurable and
  validate with logged similarity scores before fixing a value.
- **Embedding payload size:** embeddings are ~192–512 floats per speaker per
  chunk. Negligible over local HTTP, but send as compact `float32` lists.
- **Backward compatibility:** the fallback to label matching ensures no
  regression when embeddings are absent.

---

## 9. Validation Strategy

1. **Diagnostic build first (recommended):** log the cosine similarity for every
   segment against the locked primary *before* enforcing the discard, to confirm
   same-speaker vs different-speaker scores separate cleanly. Use that data to
   set the threshold.
2. Two-speaker test: primary asks a question, a bystander interjects mid-session.
   Expect `[SPEAKER-FILTER] ... != primary → DISCARD` for the bystander and the
   RAG transcript to exclude their text.
3. Single-speaker regression: confirm the primary's full transcript is retained
   (no false discards) across multiple chunks.
4. Diarization-disabled regression: confirm flat-text path is unaffected.

---

## 10. References

- `docs/speaker-separation-lld.md` — original low-level design.
- `docs/speaker-separation-design.md` — original high-level design.
- pyannote.audio 4.0.4 — `DiarizeOutput.speaker_embeddings`.
- Source service: `edge-ai-suites/education-ai-suite/smart-classroom`.
