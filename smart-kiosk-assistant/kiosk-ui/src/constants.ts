export const constants = {
  TITLE: 'Kiosk Voice Assistant',
  COPYRIGHT: '© 2026 Intel Corporation. All rights reserved.',
  VERSION: '2026.1.0',
};

// Endpoint paths (proxied to backends by Vite in dev / nginx in prod).
export const endpoints = {
  // kiosk-core
  startStream: '/api/v1/sessions/start-stream',
  pushAudio: (sid: string) => `/api/v1/sessions/${sid}/audio`,
  endAudio: (sid: string) => `/api/v1/sessions/${sid}/audio/end`,
  pollSession: (sid: string) => `/api/v1/sessions/${sid}`,
  sessionAudioFile: (sid: string, filename: string) =>
    `/api/v1/sessions/${sid}/audio/${encodeURIComponent(filename)}`,
  // ordering
  products: '/api/v1/products',
  order: (orderId: number) => `/api/v1/orders/${orderId}`,
  currentOrder: (userId: string) =>
    `/api/v1/users/${encodeURIComponent(userId)}/orders/current`,
  upsell: '/api/v1/upsell',
  // rag-service (proxied under /rag)
  ragContext: '/rag/api/v1/context',
  ragContextFile: '/rag/api/v1/context/file',
  ragModelInfo: '/rag/api/v1/model-info',
  ragPerformance: '/rag/api/v1/performance',
  // audio-analyzer (ASR, proxied under /asr)
  asrModelInfo: '/asr/v1/model-info',
  asrPerformance: '/asr/v1/performance',
  // text-to-speech (proxied under /tts)
  ttsModelInfo: '/tts/v1/model-info',
  ttsPerformance: '/tts/v1/performance',
  // metrics-collector (proxied under /metrics-svc)
  metrics: '/metrics-svc/metrics',
  // pipeline latency (kiosk-core)
  pipelineLatest: '/api/v1/pipeline/latest',
  // Ring buffer of recent turn traces — powers the per-request voice-to-voice
  // history under the AI Inference Pipeline panel.
  pipelineRecent: '/api/v1/pipeline/recent?n=8',
  // identity (biometric auth, proxied to kiosk-core; feature-flag gated)
  identityEnabled: '/api/v1/identity/enabled',
  identityChallenge: '/api/v1/identity/challenge',
  identityVerify: '/api/v1/identity/verify',
  identityRegister: '/api/v1/identity/register',
};

// Tuning constants (mirror kiosk_core config defaults).
export const tuning = {
  // Upload cadence only — how often buffered audio is POSTed while recording.
  // Kept short so audio reaches the backend promptly; it does NOT bound the
  // ASR chunk (see asrChunkSeconds).
  //
  // Lowered from 2.5s: at 2.5s, trailing silence sat buffered on the client
  // for up to a full 2.5s before the backend ever saw it, so the endpoint/
  // completeness shortcut (kiosk_core config.py DEFAULT_ENDPOINT_SHORT_SECONDS)
  // could never fire in time — the server can't detect silence it hasn't
  // received yet. 0.5s matches kiosk-voice-lab-main's rolling-snapshot cadence
  // (pipeline/config.py tick_s=0.4) closely enough to let the backend's
  // quiet-mode preview acceleration (DEFAULT_REALTIME_PREVIEW_COMMIT_QUIET_SECONDS,
  // 0.15s) actually have fresh audio to act on.
  chunkSeconds: 0.5,
  // Max seconds of speech the backend accumulates before force-flushing to
  // Whisper, sent as chunk_seconds on session start.
  //
  // This was previously taken from chunkSeconds (2.5s), which made the upload
  // cadence double as a hard ASR boundary: speech was sliced every 2.5s
  // regardless of word boundaries. Measured live, "one Aloo Tikki Burger and
  // two Classic Chicken Burger" was cut into "I would like to have one" /
  // "and 2,000" / "Classic Chicken Burger" — the item name straddled a cut and
  // was lost. Whisper needs whole phrases for context; a fragment decodes into
  // whatever is phonetically closest.
  //
  // 6.0s matches KIOSK_CORE_CHUNK_SECONDS. It is only a ceiling — the backend
  // still flushes earlier at natural pauses (adaptive flush), so ordinary
  // speech latency is unchanged and this bites only on continuous speech.
  asrChunkSeconds: 6.0,
  // Push-to-talk only. When the whole recording is uploaded as a single chunk
  // on Stop, the backend must not re-split it, so its chunk cap is raised to
  // the maximum the request model allows (chunk_seconds <= 30) and the session
  // cap is set to the SAME value. Making them equal is what guarantees one
  // chunk: a recording can never outlive the session cap, so it can never
  // reach the chunk cap either, and the explicit end-of-stream signal from the
  // Stop button stays the only thing that ends a turn.
  singleChunkMaxSeconds: 30.0,
  // Trailing-silence endpoint for push-to-talk, at the model's ceiling
  // (silence_timeout_seconds <= 10). It is deliberately far longer than any
  // pause inside a sentence, so it cannot cut the customer off; Stop ends the
  // turn. It still exists as a backstop for a recording that is pure silence.
  singleChunkSilenceSeconds: 10.0,
  // Trailing silence that ends the turn. THIS OVERRIDES the backend default
  // (KIOSK_CORE_SILENCE_TIMEOUT_SECONDS) — it is sent as
  // silence_timeout_seconds on session start, so this constant is the value
  // that actually decides when the customer gets cut off.
  //
  // Was 0.65s, which is shorter than an ordinary mid-sentence pause: a
  // customer glancing at the menu ("I would like to order one … Classic
  // Chicken Burger") had the turn ended during the hesitation. Measured live,
  // every turn captured only 1.8-2.5s of audio and Whisper received truncated
  // sentences — it even emitted a literal "..." for one, and guessed a wrong
  // final word on another, because the item name was never recorded. That
  // looked like an ASR accuracy problem but was purely an endpointing one.
  //
  // Must stay strictly greater than the backend's adaptive flush pause,
  // otherwise the endpoint always fires first and the pre-warm flush that
  // hides ASR latency can never run.
  //
  // 1.1s (kiosk-voice-lab-main parity, endpoint_long_ms): lowered from 1.5s
  // to match the lab's "unfinished" wait. Mirrors
  // KIOSK_CORE_SILENCE_TIMEOUT_SECONDS in kiosk_core/config.py — keep both in
  // sync.
  silenceTimeoutSeconds: 1.1,
  sampleRate: 16000,
  pollIntervalMs: 350,
  perfRefreshMs: 10000,
  maxHistoryTurns: 4,
  userId: 'kiosk-user',
};

// Built-in sample knowledge bases (served from /samples).
export const sampleKnowledgeBases: { label: string; file: string }[] = [
  { label: 'QuickBite (QSR)', file: 'QuickBite-M.md' },
  { label: 'MegaRetail (Retail Store)', file: 'MegaRetail-M.md' },
  { label: 'SkyJet (Airline)', file: 'SkyJet-S.md' },
];
