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
};

// Tuning constants (mirror kiosk_core config defaults).
export const tuning = {
  chunkSeconds: 5.0,
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
