# Audio Analyzer — Diarization Improvement Plan

**Date:** 2026-07-01  
**Author:** Copilot  
**Status:** Implemented (2026-07-03) — see "Implementation Notes" at the end for
how the final design differs from the original proposal below.

---

## 1. Problem Statement

When two people speak within the same audio chunk (5–15 s), the current pipeline only
surfaces one speaker's text to the kiosk. The other voice is silently discarded.
Additionally, cross-chunk speaker labels are unstable: pyannote resets its anonymous
IDs (`SPEAKER_00`, `SPEAKER_01`, …) on every chunk, so the same physical person can
receive different labels in consecutive chunks. This causes the kiosk-core speaker
filter to lock onto a stale label and either drop the customer's own speech or keep
a staff member's voice.

### Root Causes (two layers)

#### Layer 1 — Inside `audio-analyzer` (pyannote)

`PyannoteDiarizer.diarize()` calls `output.exclusive_speaker_diarization`, which is
pyannote's **exclusive / non-overlapping** mode. When two people speak simultaneously
the dominant voice wins and the overlap is discarded entirely. Each Whisper ASR segment
is then assigned a speaker by **midpoint lookup**:

```python
mid = (sent["start"] + sent["end"]) / 2.0
for turn in speaker_turns:
    if turn["start"] <= mid <= turn["end"]:
        speaker = turn["speaker"]
        break
```

If the segment midpoint falls in a gap or on the boundary of two turns the speaker is
`UNKNOWN`. If the segment spans two speaker turns the midpoint arbitrarily picks one.

No `min_speakers` / `max_speakers` constraint is passed to pyannote, so the clustering
algorithm is free to produce 3–5 phantom speakers on short noisy audio.

#### Layer 2 — Inside `kiosk-core` (speaker filter)

`_filter_target_speaker()` in `audio_session.py` locks onto the **first meaningful
KNOWN speaker label** as `primary_speaker_id` and discards every other speaker for the
entire session. Because pyannote resets its anonymous labels per chunk, the same
physical customer can appear as `SPEAKER_01` in chunk 2 after being `SPEAKER_00` in
chunk 1 — causing their own speech to be discarded as a "secondary voice".

There is no embedding-based cross-chunk identity matching. The only fallback is a
keyword-overlap semantic scorer, which fires only when the primary is completely silent
for a full chunk.

---

## 2. Scope

| Service | Files Affected |
|---|---|
| `audio-analyzer` | `components/asr/diarization/pyannote_diarizer.py` |
| `audio-analyzer` | `components/asr_component.py` |
| `audio-analyzer` | `config.yaml` |
| `kiosk-core` | `kiosk_core/audio_session.py` |

---

## 3. Proposed Solution — Two Phases

### Phase 1 — Fix inside `audio-analyzer`

#### 1a. Switch from `exclusive` to full `Annotation` output

`exclusive_speaker_diarization` collapses overlapping speech. Switching to the full
`Annotation` object preserves overlapping turns and gives the caller a complete picture.

```python
# current
diarization = output.exclusive_speaker_diarization

# proposed
diarization = output   # full pyannote Annotation — includes overlapping regions
```

#### 1b. Expose all speaker turns that overlap each time window

Instead of returning a single `{start, end, speaker}` per turn, return every turn that
overlaps with the chunk so that `asr_component.py` can make a more informed assignment
decision.

Response shape is unchanged — we still emit `{start, end, speaker}` items — but now
overlapping regions produce two entries with different `speaker` labels, which callers
can inspect.

#### 1c. Pass `min_speakers` / `max_speakers` config hints to the pipeline

For a kiosk (1 customer + at most 1 staff member), constraining the speaker count
dramatically reduces pyannote clustering errors on short audio:

```python
# pyannote_diarizer.py
output = self.pipeline(
    audio_input,
    min_speakers=self.min_speakers,   # from config, default 1
    max_speakers=self.max_speakers,   # from config, default 2
)
```

```yaml
# config.yaml — new fields under models.diarization
diarization:
  provider: "huggingface"
  name: "pyannote/speaker-diarization-community-1"
  device: CPU
  models_base_path: "models"
  min_speakers: 1      # NEW — minimum expected speakers per chunk
  max_speakers: 2      # NEW — maximum expected speakers per chunk
```

**Risk:** near-zero. If audio is silent or single-speaker, pyannote correctly returns
one label even with `min_speakers=1`.

#### 1d. Assign speaker by maximum overlap area, not midpoint

The midpoint heuristic breaks when a Whisper segment spans two speaker turns. Replace
it with the turn that has the **greatest time overlap** with the segment interval:

```python
# current (midpoint)
mid = (sent["start"] + sent["end"]) / 2.0
for turn in speaker_turns:
    if turn["start"] <= mid <= turn["end"]:
        speaker = turn["speaker"]
        break

# proposed (max overlap)
best_speaker, best_overlap = None, 0.0
for turn in speaker_turns:
    overlap = min(sent["end"], turn["end"]) - max(sent["start"], turn["start"])
    if overlap > best_overlap:
        best_overlap = overlap
        best_speaker = turn["speaker"]
speaker = best_speaker
```

**Risk:** near-zero. Strictly more correct than midpoint.

#### 1e. (Optional) Expose per-speaker embeddings in the response

Pyannote internally computes speaker embeddings during clustering. Expose the mean
embedding for each speaker label as an optional field in the diarization response so
kiosk-core can do cross-chunk identity matching without loading a second model.

Proposed addition to the segment dict:

```json
{
  "speaker": "SPEAKER_00",
  "start": 0.5,
  "end": 3.2,
  "embedding": [0.12, -0.04, ...]
}
```

---

### Phase 2 — Fix inside `kiosk-core`

#### 2a. Replace label-based lock with embedding-based cross-chunk identity

Add a `primary_speaker_embedding` field to `BaseAudioSession` alongside the existing
`primary_speaker_id`:

```python
# current
self.primary_speaker_id: str | None = None   # anonymous label only

# proposed
self.primary_speaker_id: str | None = None
self.primary_speaker_embedding: np.ndarray | None = None   # 192-dim cosine vector
```

Lock-on logic becomes:

1. First chunk with a meaningful segment → compute/store embedding as primary.
2. Every subsequent chunk → cosine-compare new segment embedding against stored primary.
3. Similarity ≥ threshold (e.g. 0.75) → confirmed same person → keep / promote.
4. Similarity < threshold → different voice → discard (or trigger semantic fallback).

This makes the primary speaker **identity-stable** across chunk boundaries even though
pyannote resets its anonymous labels every chunk.

#### 2b. Consume `embedding` field from audio-analyzer response

If Phase 1e is implemented, kiosk-core reads the `embedding` from the segment dict
directly — no second embedding model needed in kiosk-core. If not implemented, a
lightweight fallback (e.g. SpeechBrain ECAPA-TDNN already available in the model
stack) can be used.

#### 2c. Config-driven similarity threshold

Add `DEFAULT_SPEAKER_SIMILARITY_THRESHOLD` to `kiosk_core/config.py` (env var:
`KIOSK_CORE_SPEAKER_SIMILARITY_THRESHOLD`, default `0.75`) so the threshold can be
tuned per deployment without code changes.

---

## 4. Change Summary

| # | File | Change | Risk | Priority |
|---|---|---|---|---|
| 1c | `audio-analyzer/config.yaml` | Add `min_speakers: 1`, `max_speakers: 2` under `diarization` | Low | **High — Quick win** |
| 1c | `pyannote_diarizer.py` | Pass `min_speakers`/`max_speakers` to `self.pipeline()` | Low | **High — Quick win** |
| 1d | `asr_component.py` | Replace midpoint lookup with max-overlap assignment | Low | **High — Quick win** |
| 1a | `pyannote_diarizer.py` | Use full `Annotation` output instead of `exclusive_speaker_diarization` | Medium | Medium |
| 1b | `pyannote_diarizer.py` | Emit overlapping turn entries in response | Medium | Medium |
| 2a | `kiosk_core/audio_session.py` | Embedding-based cross-chunk speaker re-ID | Medium | Medium |
| 2b | `audio_session.py` | Consume `embedding` from segment response | Low | Medium |
| 1e | `pyannote_diarizer.py` | Extract and return per-speaker embedding | High | Optional |
| 2c | `kiosk_core/config.py` | `DEFAULT_SPEAKER_SIMILARITY_THRESHOLD` env var | Low | Optional |

---

## 5. Recommended Implementation Order

```
Step 1 (Quick wins — no new models, minimal risk)
  → 1c: min_speakers/max_speakers in config + pyannote call
  → 1d: max-overlap speaker assignment in asr_component.py

Step 2 (Correct the architecture)
  → 1a + 1b: full Annotation output + overlapping turns in response
  → 2a + 2b: embedding-based cross-chunk identity in kiosk-core
  → 2c: similarity threshold config

Step 3 (Optional / advanced)
  → 1e: expose raw pyannote embeddings in the segment response
```

---

## 6. Validation

| Check | How to Verify |
|---|---|
| Two speakers in same chunk both appear in logs | `docker logs audio-analyzer \| grep DIARIZATION` — two distinct `SPEAKER_XX` entries per chunk |
| Same customer keeps primary across chunks | `docker logs kiosk-core \| grep SPEAKER-FILTER` — `primary_speaker_id` unchanged across chunk boundaries |
| Staff voice correctly discarded | Speak as staff while customer is silent → `DISCARD (secondary voice)` in logs |
| `min_speakers` reduces phantom speakers | Single-person audio produces exactly 1 speaker label, not 3–5 |
| Max-overlap correct on boundary segments | Segment spanning two turns → speaker = turn with longer overlap |

---

## 7. References

- `audio-analyzer/components/asr/diarization/pyannote_diarizer.py` — current diarizer
- `audio-analyzer/components/asr_component.py` — midpoint assignment (line ~143)
- `audio-analyzer/config.yaml` — diarization section
- `kiosk-core/kiosk_core/audio_session.py` — `_filter_target_speaker()` method
- pyannote-audio docs: https://github.com/pyannote/pyannote-audio

---

## 8. Implementation Notes (2026-07-03)

The final implementation deviates from Section 3/Phase 2 above at the
explicit direction that kiosk-core must not load any additional model:
**audio-analyzer resolves the stable primary speaker itself** and hands
kiosk-core a simple `is_primary` boolean per segment. kiosk-core no longer
holds any cross-chunk speaker state.

What was implemented, in order:

1. **Quick wins (audio-analyzer)** — `min_speakers`/`max_speakers` (1/2) passed
   to the pyannote pipeline call; midpoint speaker assignment in
   `asr_component.py` replaced with max-overlap assignment.
2. **`components/asr/diarization/speaker_identity.py` (new)** — thread-safe,
   in-memory `SpeakerIdentityStore`, keyed by `session_id`. Reuses the speaker
   embeddings pyannote already computes during clustering
   (`DiarizeOutput.speaker_embeddings`, confirmed public API in
   pyannote-audio 4.0.4) — **no additional model is loaded**. Locks onto the
   longest-speaking label in the first chunk that meets
   `lock_min_duration_sec`, then cosine-compares every later chunk's speaker
   embeddings against the frozen primary embedding
   (`similarity_threshold`, default 0.75). Idle sessions are evicted after
   `session_ttl_seconds` (default 1800s).
3. **`pyannote_diarizer.py`** — `diarize()` now returns
   `(turns, label_embeddings)`; `label_embeddings` maps each local speaker
   label to its embedding via `output.speaker_diarization.labels()` /
   `output.speaker_embeddings`.
4. **`asr_component.py`** — calls `SpeakerIdentityStore.resolve()` per chunk
   and stamps each transcribed segment with `is_primary: bool` alongside the
   existing `speaker` label.
5. **`config.yaml`** — new `diarization.min_speakers`, `max_speakers`, and
   `diarization.identity.{enabled, similarity_threshold,
   lock_min_duration_sec, session_ttl_seconds}`.
6. **`kiosk_core/audio_session.py`** — `_filter_target_speaker()` rewritten to
   trust `segment["is_primary"]` directly; removed `primary_speaker_id`,
   `pending_segments`, the lock-on/buffering state machine, and the
   `_meaningful_char_count()` helper. The domain-keyword semantic fallback
   (for chunks where the primary was silent) is unchanged and still lives in
   kiosk-core, since it is kiosk-specific vocabulary, not audio science.
   The `primary_speaker_id` field was also removed from the session snapshot
   API and `kiosk-ui/src/types.ts` (it was unused by the UI).

Tests added:
- `audio-analyzer/tests/test_speaker_identity.py` (7 cases: lock-on,
  cross-chunk identity stability despite label reset, dissimilar-embedding
  rejection, session isolation, TTL eviction, reset).
- `audio-analyzer/tests/test_pyannote_diarizer.py` updated for the new
  `(turns, label_embeddings)` return signature and `min/max_speakers` kwargs.
- `smart-kiosk-assistant/tests/unit/test_speaker_filter.py` (6 cases) for the
  simplified `_filter_target_speaker()`.

Items from the original Section 3 that were **not** implemented (left for a
future iteration, low priority): 1a/1b full non-exclusive `Annotation` output
to recover simultaneous overlapping speech — the current fix targets the
cross-chunk identity instability, which was the primary reported issue;
overlapping-speech recovery is a separate, lower-frequency problem.

