# PR #112 — Response to Josh's Review (Team Briefing)

**PR:** [intel-retail/voice-enabled-interactions#112](https://github.com/intel-retail/voice-enabled-interactions/pull/112) — "Performance improvement"
**Reviewer:** Josh Cork (`jcork-intel`) — `CHANGES_REQUESTED`
**Review submitted:** 2026-10-02, against commit `be237e8`
**Branch:** `performance-improvement` · **HEAD at time of writing:** `c62c251`
**Commits landed after the review:** 27 (`be237e8..c62c251`)

This document maps every item Josh raised to the commit(s) that address it, explains the
*logic* of each change (so it can be re-explained to the team), and flags the two items that
are **not yet safely addressed on this branch**.

---

## 1. At-a-glance status

Legend: ✅ addressed · ⚠️ partial / needs decision · ❌ regressed or open

### Blocking (1–6)

| # | Item | Status | Commit(s) |
|---|------|--------|-----------|
| 1 | V2V anchored on customer's **first** word, not last | ✅ | `1f70eae` |
| 2 | Opener corrupts first-audio metric; adopt agreed latency vocabulary; retire "time to first audio" | ✅ | `1f70eae`, `039922d`, `da475ee`, `c62c251` |
| 3 | Default stack can't transcribe (`language=en` + English-only model; `.env` var not passed through) | ✅ | `21eaf49` |
| 4 | Build depends on **unmerged** code in two repos (edge-ai-libraries, performance-tools) | ✅ (by design) | `37996ed` (pinned + documented) — **audio-analyzer/TTS will ship as registry images, not a build reference** (see §3) |
| 5 | `make benchmark` aborts on shipped `.env` (`ASR_PREVIEW_DEVICE=${ASR_DEVICE:-CPU}`) + no submodule | ✅ | `37996ed` |
| 6 | 3 rag-service tests fail; CI never ran `rag-service/tests` | ✅ | `8252d9d` |

### Correctness (7–12)

| # | Item | Status | Commit(s) |
|---|------|--------|-----------|
| 7 | `price_guard` rewrites item counts ("2 burgers" → "338 burgers") | ✅ | `9a67ed0` |
| 8 | Opener edge cases (plays on agent fail; `None`-cached forever; skips trim/gain) | ✅ | `d78d440` |
| 9 | Silero VAD falls back to RMS silently; trace doesn't record which ran | ✅ | `5e281ec` |
| 10 | `ADAPTIVE_FLUSH_PAUSE_SECONDS=0.30` below the documented 0.50 floor | ✅ | `52426c2` |
| 11 | Hands-free mode unwired; keep it supported & measure latency in it | ✅ | `d65f25a` (hands-free restored as the default on both screens); latency measured via the speaker clock (#2) |
| 12 | Conflicting defaults (`ASR_DEVICE`, `ENDPOINT_STABLE_SECONDS`, `RELEASE_TAG`) | ✅ | `88bf6e5` |

### Benchmark harness (13–21)

| # | Item | Status | Commit(s) |
|---|------|--------|-----------|
| 13 | `consolidated_metrics.csv` one row per session, no aggregate | ✅ | `8299017` |
| 14 | Makefile warm-up hides failures; `V2V_WARMUP_RUNS=0` doesn't disable it | ✅ | `9195a47` |
| 15 | Four turns can't support a p95; run count set 3 different ways | ✅ | `367bba5` |
| 16 | Manifest records *requested*, not *actual*, settings; stale numbers on failure | ✅ | `9195a47` |
| 17 | 9/10 profiles use SpeechT5/Ryan but compose hardcodes kokoro/am_michael | ✅ | `88bf6e5` |
| 18 | Utilisation averaged over whole stack lifetime, not the measured turns | ✅ | `c37e142` |
| 19 | Split the ~1,900-line device-matrix harness into its own PR | ⚠️ | Product/process decision — **not yet split** |
| 20 | Non-writable `results/` (uid 1000 vs host uid) fails silently | ✅ | `36a7612` |
| 21 | Benchmark always passes `--explicit-end-mark`, so never measures endpointing | ✅ | `353031f`, `039922d` |

### Cleanup & tests (22–27)

| # | Item | Status | Commit(s) |
|---|------|--------|-----------|
| 22 | Prompt rewrite lives in a module nothing imports | ✅ | `2188b4d` |
| 23 | `gradio`/`plotly` still installed; perf-tools installed from `@main` | ✅ | `81c1529` |
| 24 | References to deleted files (`*_no_adk.py`, `baseline.json`, …) | ✅ | `81940dd` |
| 25 | New code without tests (`directive_mode`, `for_speech`, `RealtimeAnalyzerClient`, opener) | ✅ | `7f39c6c`, `d78d440` |
| 26 | Dashboard TTS card reads 0 ms; "p95" is the max over a short tail | ✅ | `099e099`, `1f70eae` |
| 27 | Document host prereqs (`python3-venv`, `ffmpeg`) and `--recurse-submodules` | ✅ | `37996ed`, `da475ee` |

**Score: 25 ✅ addressed · 1 ⚠️ decision (19 device-matrix split) · 1 by product decision (4 registry images).**

---

## 2. What each change does and why (the logic)

### Blocking

**#1 — V2V started at the first word (`1f70eae`).**
`_t_last_speech_frame_sent` was stamped on the first speech frame, but every later update sat
inside `if self._streaming_active:`. Streaming is **off** by default, so on the shipped path the
anchor never moved past the customer's *first* word — every turn was over-reported by roughly the
length of the utterance (worse for exactly the long, multi-item orders that matter most). The stamp
now updates on **every** speech frame regardless of streaming mode, renamed `_t_last_speech_frame`.
Pinned by `test_last_word_anchor.py`, which fails if the gating is ever restored.

**#2 — Opener corrupted the headline metric; latency vocabulary retired/aligned (`1f70eae`, `039922d`, `da475ee`, `c62c251`).**
Three defects: (a) the clock started at the first word (see #1); (b) it *stopped* when the cached
"One moment." opener WAV was **copied**, before the agent even ran — so the number measured
end-of-turn detection + a file copy, not the pipeline; (c) spans mixed audio-domain
`silence_run_seconds` with wall-clock instants, producing negative values that were silently
clamped. Fixes: a new wall-clock `_t_endpoint_decision` stamp makes all three headline spans pure
wall-clock deltas that reconcile exactly — **`voice_to_voice_ms == endpointing_delay_ms +
processing_latency_ms`**. Every turn now records `first_audio_was_opener`, and `_emit_vlm_metrics`
publishes the **answer-anchored** instant. The agreed vocabulary is adopted with **no compatibility
aliases** (so nothing can keep reading a retired field): voice-to-voice, endpointing delay,
processing latency, transcription latency, LLM TTFT, TTS TTFB. "Time to first audio" is retired.
`039922d` additionally anchors benchmark V2V on **where speech ends in the clip** (ffmpeg
`silencedetect`) instead of the VAD, since endpointing is part of what's being measured; every KPI
carries an `anchor` field. `c62c251` disables the opener by default so the shipped metric reflects
the pipeline. `da475ee` writes the vocabulary down once in `docs/user-guide/measuring-latency.md`.

**#3 — Default stack couldn't transcribe (`21eaf49`).**
kiosk-core sent `language=en` on every file-path request, but the shipped model
`distil-whisper/distil-small.en` is English-only; audio-analyzer turns a language value into an
`<|en|>` token, which English-only checkpoints reject → HTTP 500, empty transcripts. `.env`'s
`KIOSK_CORE_ASR_LANGUAGE=` didn't help because compose uses an explicit `environment:` allowlist and
never passed the variable through. Two fixes: (1) compose now passes `KIOSK_CORE_ASR_LANGUAGE`
through, defaulting it **empty** using `${VAR-}` (no colon) so an explicitly-empty value survives;
(2) `AnalyzerClient` self-heals — on a language-rejection it latches a process-wide flag, retries
once without the language token, and warns only the first time.

**#4 — Build depends on unmerged upstream (`37996ed`) — RESOLVED BY DESIGN.**
Josh's ask was "land both upstream first, **or** pin both to a commit and document it." Both resolve
the same way: **audio-analyzer and text-to-speech are merging to upstream `main` and Smart Kiosk will
consume them as published registry images, not as a source build.** docker-compose.yml's `build:`
reference to `../edge-ai-libraries` is a local-development convenience only; the shipped/CI path pulls
the released images (`image:` + `REGISTRY`). So once the two services land on main, there is no
build-time dependency on their source and nothing to pin. In the meantime `37996ed` already made the
source path honest: `make build` uses `set -e` (a failed image build no longer prints "✓ Images ready"
and exits 0), `make update-submodules` no longer runs `--remote`, and `build-from-source.md` documents
the pins for anyone building from source. **Net: not a merge blocker — the production stack uses images.**

**#5 — `make benchmark` aborted on shipped `.env` (`37996ed`).**
`.env.example` set `ASR_PREVIEW_DEVICE=${ASR_DEVICE:-CPU}`. A `.env` file is not a shell, so the
validator read that string verbatim and rejected it. The line is now commented out, letting the
preview pool inherit `ASR_DEVICE` through compose's own nested fallback (where expansion really
happens); the validator also grew a dedicated error for any value still containing `${`. Combined
with the `--recurse-submodules` doc fix (#27), a documented clone now reaches the orchestrator.

**#6 — rag-service tests broken + not in CI (`8252d9d`).**
`test_make_mcp_callable_*` broke because `dry_run` is now enforced server-side for speculative drafts
(so every mutating call carries `dry_run=False`) — assertions updated to the **intended new
contract**. `test_speakable_tools_set` broke because two read tools were added to the speakable set;
rather than re-listing names, a new invariant test asserts `_READ_TOOLS <= SPEAKABLE_TOOLS`. A
`rag-service-unit-tests` CI job was added (runs in parallel with tier1, minimal deps). Verified
366 passed.

### Correctness

**#7 — price_guard rewrote item counts (`9a67ed0`).** `_TOTAL_MENTION_RE` took the first number
after "total", so "Your total for 2 burgers is ₹338" became "…338 burgers…". The pattern now
requires a connector ("is", "comes to", …) or a currency symbol immediately before the captured
number. Regression tests pin both directions (what must be rewritten vs. item counts left alone).

**#8 — opener edge cases (`d78d440`).** A failed synthesis was cached as `None` for the process
lifetime (one bad round-trip at startup disabled the opener forever, silently) — now remembered for
`KIOSK_CORE_OPENER_RETRY_SECONDS` (60 s) then retried, with `opener_failed` recorded in the trace and
the partial file deleted. The opener was a plain file copy, skipping the trim/gain every other
segment gets — now normalised once at render time. If the agent call then fails, the turn closes
with a non-committal apology (queued before TTS drains) so the customer never hears "One moment."
then silence.

**#9 — silent VAD fallback (`5e281ec`).** Silero falls back to RMS when the ONNX model/onnxruntime is
missing or the sample rate is unsupported, which changes endpointing — and nothing recorded which
ran (CI tier1 runs before the model download, so it exercises RMS). Adds `WallTimes.vad_backend`, set
on every branch and logged once per session; selection extracted to `_init_vad_backend` for direct
testing.

**#10 — adaptive-flush floor (`52426c2`).** compose/`.env.example` shipped `0.30`, below the `0.50`
floor config.py documents as validated; `0.30` clears the chunk mid-phrase and Whisper
hallucinates/truncates the tail. Restored to `0.50`. Confirmed on the 88-turn sweep: a turn that
transcribed as "Wait, actually." at `0.30` returns the full sentence at `0.50`, V2V p95 5618 → 5417 ms.

**#11 — hands-free mode (`fe22572`, briefly reverted by `c62c251`, restored by `d65f25a`) — RESOLVED.**
Hands-free is the default conversation mode on both the customer kiosk screen (`CustomerApp` →
`AskBar`, `onStart={startConversation}`) and the chat screen. Tapping the mic opens a persistent
full-duplex WebSocket (`voiceSocket.ts`): the browser streams binary PCM frames and the turn ends on
server-side endpointing, with a barge-in tap and an explicit "End conversation" control. The ~1 s
post-speech wait that previously hurt the experience is no longer hidden from the headline number —
the speaker clock (#2) now stops voice-to-voice at the first sound at the speaker and splits the total
into *endpointing delay* vs *processing latency*, so hands-free latency is measured honestly in the
benchmark (which drives the same WebSocket path). Josh's request — keep hands-free supported and
measure latency in it — is satisfied.

**#12 — conflicting defaults (`88bf6e5`).** `ASR_DEVICE` (compose CPU vs. config.yaml GPU) → GPU
everywhere (GPU is the only device `distil-small.en` is confirmed correct on).
`KIOSK_CORE_ENDPOINT_STABLE_SECONDS` (compose 0.2 vs. code 0) → aligned (and later set to 0.2 for
correctness, see below). `RELEASE_TAG` (`.env.example` `latest` vs. Makefile `2026.1.0`) → `.env.example`
now ships `2026.1.0` (Makefile's `?=` can't override a `-include`d `.env`).

### Benchmark harness

**#13 — one row per session (`8299017`).** `_emit_vlm_metrics` stamped each turn with the per-turn
`session_id`; the consolidator groups by `{application}_{id}` → 88 rows of "1 total call". The id now
uses a stable `VLM_METRICS_STREAM_ID`, collapsing to the single aggregate row the consolidator is
designed to emit. Per-turn detail still lives in the JSON report + `/api/v1/pipeline/latest`.

**#14 & #16 — honest warm-up and manifest (`9195a47`).** The Makefile ran its own warm-up (brought the
stack up/down a second time, warmed a process then discarded, `V2V_WARMUP_RUNS=0` didn't disable the
orchestrator's own warm-up, failures went to `/dev/null || true`) — now delegated to the
orchestrator's `--warmup_runs`. The manifest recorded *requested* queue/diarization settings although
the orchestrator forces both off — now records **effective** values, keeps `*_requested` alongside,
and lists overrides. A failed run left a fresh timestamp next to the previous run's numbers — the
manifest is now written `status: started` and only flipped to `completed` by `finish_run_manifest.py`
after exit 0, so a lingering "started" marks its neighbours stale.

**#15 — enough turns for p95 (`367bba5`).** Defaults gave 4 turns from one conversation (p95 = slowest
turn); run count was set 3 ways (Makefile 1, orchestrator 12, child 3). `V2V_SCRIPT` now defaults to
`all` (11 turns/pass over varied conversations) and `V2V_RUNS` to 8 → 88 turns; child default moved
to 8 so all three agree.

**#17 — TTS voice/model hardcoded (`88bf6e5`).** compose hardcoded `KIOSK_CORE_TTS_VOICE: am_michael`
and `KIOSK_CORE_TTS_MODEL: kokoro` (what kiosk-core *requests*), but 9/10 benchmark profiles switch the
service to SpeechT5/Ryan — so the client asked for a Kokoro voice the service couldn't serve and every
TTS request failed. Both are now parameterised from the same variables that configure the service
(`${TTS_SPEAKER:-am_michael}`, `${TTS_MODEL:-kokoro}`); shipped defaults unchanged.

**#18 — utilisation window (`c37e142`).** CPU/GPU/NPU averages spanned everything the collector saw
(image pulls, health wait, discarded warm-up, teardown). The orchestrator now records the measured
span and `trim_metrics_to_window.py` narrows each counter file to it before parse/consolidate. Since
no counter format carries an epoch timestamp, each file is anchored by its own mtime (hence
`sync-metrics-into-results` now copies with `cp -p`). Verified against a real 88-turn run.

**#19 — split the device-matrix harness (⚠️ decision).** Josh suggested the ~1,900-line matrix land as
its own PR so #112 can merge on the latency fixes. **Not yet actioned** — this is a
product/process call, not a code fix.

**#20 — non-writable `results/` (`36a7612`).** kiosk-core runs as uid 1000; the bind mount is owned by
whoever cloned → writes fail best-effort, run looks healthy but reports zero transactions.
`make setup-dirs` now checks ownership and widens mode when uid ≠ 1000; kiosk-core checks once at
startup and logs the path/uid/fix; the per-turn swallow warns once per process.

**#21 — benchmark skipped endpointing (`353031f`, `039922d`).** The orchestrator always passed
`--explicit-end-mark`. The performance-tools submodule was advanced to the commit that stops forcing
it, so `make benchmark` now measures real endpointing delay; the clip anchor (#2/`039922d`) makes the
detector's own error visible as `early_commit_count`.

### Cleanup & tests

**#22 (`2188b4d`)** — the condensed prompt rewrite lived in `rag-service/agentic/ordering_agent.py`,
which nothing imports (the live agent is `plugins/kiosk/ordering_agent.py` via `load_agent_factory`).
Reverted to the original prompt (the experiment also dropped guardrails) and added a warning header
naming the live module; the matching `prompt_description` fix was kept.

**#23 (`81c1529`)** — `gradio`/`plotly` were only used by the deleted `gradio_app.py`; removed. Pinned
the `vlm_metrics_logger` install to a commit instead of `@main` so an upstream log-format change can't
silently break the consolidator.

**#24 (`81940dd`)** — comments in compose/`agent_endpoints.py` pointed at deleted `*_no_adk.py`; the
`/chat-no-adk` route is still used by `agent_latency_benchmark.py`, so references were corrected rather
than the route removed. Updated the documented perf-tools pin.

**#25 (`7f39c6c`, `d78d440`)** — tests for the untested new code: `parse_directive` (28 tests,
all-or-nothing rejection pinned clause by clause), `for_speech` (18, rule ordering),
`RealtimeAnalyzerClient` (20, via event handlers), opener edge cases (`d78d440`). The additional
endpoint commits below also add `test_endpoint_completeness.py`.

**#26 (`099e099`, `1f70eae`)** — the TTS card read sub-ms TTFB because sentence 1 is often served from
the speculative-draft/opener cache. Now `first_segment_cached` is stamped and shown as a "cached"
marker on the card; the harness reports `tts_first_segment_cached_count`. The "0 ms with opener on" and
"p95 = max of a short tail" parts were already fixed in `1f70eae` (answer-anchored card, relabelled
"max (last N)").

**#27 (`37996ed`, `da475ee`)** — `system-requirements.md` documents `python3-venv`/`ffmpeg`, clone
commands switched to `--recurse-submodules`, and the `REGISTRY=true` default (which makes `make build`
pull released images instead of building the tree) is called out.

### Bonus endpointing correctness (not numbered, support #2/#11/#21)

**`9dffe65` + `52426c2`** restore `KIOSK_CORE_ENDPOINT_STABLE_SECONDS=0.2` (two-confirmation). The
88-turn sweep showed single-confirmation (`0`) committing **55–64%** of turns *before the customer
finished speaking* — it broke confirmation outright ("Yes, that is everything. Please confirm my
order." committed as "Yes, that is everything."). This costs ~650 ms of V2V p50 (756 → 1403 ms),
accepted deliberately: a correct order at 1.4 s beats a wrong order at 0.76 s. **`5423769`** adds
transitive/clause verbs to `_INCOMPLETE_TAIL_WORDS` so "Um, I think… the Spicy Chicken…" is no longer
called a finished turn (early commits 8/88 → 0/88, p50 unchanged within noise). This is the real
reason the headline V2V moved from sub-second to ~1.4 s: **the old number was fast because it was
cutting customers off.**

### Bonus benchmark correctness — per-session harness (2026-10-15 fresh-machine finding)

**`a08c9b8`** — the scripted-conversation benchmark replayed each turn under a **fresh**
`conversation_id` (`replay_fixture` minted a new one on every call). kiosk-core clears the draft cart
whenever a turn arrives under a new conversation (`_clear_stale_cart_for_new_session`), so a script
that builds a cart over several turns and then says *"please confirm my order"* always confirmed an
**empty** cart — the premature-confirm contradiction Josh saw on the fresh machine ("Your order is
confirmed … Sorry, I couldn't confirm your order just now"). `replay_fixture` now accepts an optional
`conversation_id`; the harness threads **one id per scenario per run**, so the cart accumulates and
the confirm turn acts on the real order, while a fresh id each run keeps `--runs` repetitions
independent. `--script all` keeps each scenario its own conversation rather than bleeding one cart
across four unrelated journeys. (The kiosk side already refused to *speak* a "confirmed" claim before
the confirm tool succeeds — streaming `_SentenceGate._is_safe` + `_strip_false_confirmation` on the
final reply — so this was purely a harness defect.)

---

## 3. Product decisions (closed) and the one remaining choice

**Heads-up:** the subsections below are kept for history. Since this file was first written, **#11
has been resolved in code** — hands-free is the shipped default again (`d65f25a`) and its latency is
measured via the speaker clock (#2). The only genuinely open item is **#19** (device-matrix split).

1. **#11 hands-free → restored as default (`d65f25a`).**
   Hands-free is now the **default** conversation mode on both screens, over a persistent full-duplex
   WebSocket (binary PCM uplink; server-side endpointing; barge-in + explicit "End conversation").
   The ~1 s post-speech wait that previously made hands-free feel slow is no longer charged to the
   headline — the speaker clock (#2) splits voice-to-voice into *endpointing delay* vs *processing
   latency*, so the mode is measured honestly. This closes Josh's #11 (keep hands-free supported and
   measure latency in it).

2. **#4 upstream dependency → registry images (decided).**
   audio-analyzer and text-to-speech are merging to upstream `main`; Smart Kiosk will consume them as
   **published registry images**, not as a source-build reference. The `build:` entries in
   docker-compose.yml are a local-dev convenience only — the production/CI path pulls the images — so
   there is no merge-blocking build dependency on their source.

3. **#19 device-matrix split — still a choice.** Josh suggested the ~1,900-line matrix land as its own
   PR so #112 can merge on the latency fixes alone. Not yet actioned; a process call.

---

## 4. Commit index (chronological, oldest first)

| Commit | Date | Subject | Items |
|--------|------|---------|-------|
| `1f70eae` | 10-06 | fix(latency): V2V from last word to real answer audio | 1, 2, 26 |
| `21eaf49` | 10-06 | fix(asr): let the default English-only model transcribe | 3 |
| `37996ed` | 10-06 | build: make shipped defaults build/benchmark/document pins | 4, 5, 27 |
| `8252d9d` | 10-06 | test(rag-service): fix suite this PR broke + run in CI | 6 |
| `88bf6e5` | 10-06 | fix(config): align conflicting defaults; stop pinning TTS voice/model | 12, 17 |
| `039922d` | 10-06 | fix(benchmark): anchor V2V on where speech ends in the clip | 2, 21 |
| `367bba5` | 10-06 | fix(benchmark): run enough varied turns for a p95 | 15 |
| `a615a4b` | 10-06 | fix(ui): drop retired time-to-first-audio fallback | 2/26 |
| `353031f` | 10-06 | build(benchmark): advance perf-tools to natural endpointing | 21 |
| `52426c2` | 10-06 | fix(asr): restore documented 0.50s adaptive-flush floor | 10 |
| `9dffe65` | 10-06 | fix(endpoint): stop committing turns before customer stops | 12 (support) |
| `5423769` | 10-06 | fix(endpoint): don't call a hesitating fragment a finished turn | (support) |
| `9a67ed0` | 10-06 | fix(price-guard): stop rewriting item counts | 7 |
| `5e281ec` | 10-06 | fix(vad): record which VAD actually ran | 9 |
| `d78d440` | 10-06 | fix(opener): close the three quiet-failure paths | 8, 25 |
| `36a7612` | 10-06 | fix(benchmark): stop a non-writable results/ failing silently | 20 |
| `fe22572` | 10-06 | fix(ui): restore hands-free mode on both screens | 11 (later reverted) |
| `2188b4d` | 10-06 | revert(rag-service): undo prompt rewrite in unused module | 22 |
| `81c1529` | 10-06 | build: drop deps the deleted Gradio app needed | 23 |
| `81940dd` | 10-06 | docs: point stale references at files that still exist | 24 |
| `099e099` | 10-06 | fix(dashboard): explain sub-ms TTS TTFB | 26 |
| `9195a47` | 10-06 | fix(benchmark): honest warm-up and run manifest | 14, 16 |
| `8299017` | 10-06 | fix(benchmark): one aggregate row in consolidated_metrics.csv | 13 |
| `c37e142` | 10-06 | fix(benchmark): utilisation over the turns, not stack lifetime | 18 |
| `7f39c6c` | 10-06 | test: cover directive parsing, for_speech, realtime ASR | 25 |
| `da475ee` | 10-06 | docs: add a latency measurement guide | 2, 27 |
| `c62c251` | 10-09 | review comment fix (opener off; **push-to-talk rebind**) | 2; **re-reverts 11** |
| `9c0f11c` | 10-15 | fix(latency): stop V2V clock at first sound at the speaker | 2 (speaker clock) |
| `479d477` | 10-15 | fix(benchmark): clip-anchored headline + TTS TTFB rename | 2 |
| `d65f25a` | 10-15 | fix(ui): hands-free as the default conversation mode | 11 (restored) |
| `a08c9b8` | 10-15 | fix(benchmark): thread one conversation per scripted scenario | per-session harness |

---

## 5. Summary for the standup

- **25 of 26 numbered items are fully addressed**, each with tests or measured evidence; only #19
  (device-matrix split) remains as a process choice.
- The central narrative: the PR's headline latency looked great because it was **cutting customers
  off** (first-word anchor, opener-as-finish, single-confirmation endpointing). We fixed the
  measurement to stop at the speaker on the real answer, and fixed the endpointing to wait for the
  customer — so the honest number is ~1.4 s, not sub-second. A correct order at 1.4 s beats a wrong
  order at 0.76 s.
- **One closed product decision + two code resolutions to communicate to Josh:** (1) **hands-free is
  now the default again** (`d65f25a`) over a persistent WebSocket, with its latency measured honestly
  by the speaker clock — #11 resolved; (2) **the speaker clock stops voice-to-voice at the first
  sound at the speaker** (`9c0f11c`, `479d477`) and the benchmark headline is clip-anchored with TTS
  reported as time-to-first-byte — #2 resolved; (3) **#4 is not a blocker** — audio-analyzer/TTS will
  ship as registry images, not a source-build reference.
- Optional: split the device-matrix harness into its own PR (#19).
