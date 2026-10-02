// ── Chat ────────────────────────────────────────────────────────────────────
export type ChatRole = 'user' | 'assistant';

export interface ChatMessage {
  role: ChatRole;
  text: string;
}

// Sent to the RAG/agent for context.
export interface HistoryTurn {
  role: ChatRole;
  content: string;
}

// ── Session (kiosk-core) ────────────────────────────────────────────────────
export interface TtsAudioSegment {
  audio_file: string;
  [key: string]: unknown;
}

export interface SessionSnapshot {
  session_id: string;
  status: string; // created | running | stopping | completed | error
  transcript: string;
  response: string;
  tts_audio_segments: TtsAudioSegment[];
  tts_errors: string[];
  error: string | null;
}

export interface StartStreamResponse {
  session_id: string;
}

// ── Ordering ────────────────────────────────────────────────────────────────
export interface Product {
  product_id: string;
  name: string;
  category: string;
  price: number;
}

export interface OrderItem {
  id: number;
  order_id: number;
  product_id: string;
  product_name: string;
  quantity: number;
  price: number;
  subtotal: number;
}

export interface Order {
  order_id: number;
  user_id: string;
  status: 'draft' | 'confirmed';
  total: number;
  created_at: string;
  items: OrderItem[];
}

export interface UpsellSuggestion {
  product: Product;
  reason: string;
}

// ── KPIs ────────────────────────────────────────────────────────────────────
export interface KpiData {
  // raw merged model-info + perf; rendered defensively in ModelKpis
  [key: string]: unknown;
  perf?: Record<string, unknown>;
}

// ── Pipeline turn trace (from /api/v1/pipeline/latest) ──────────────────────
export interface PipelineWall {
  turn_total_ms: number | null;
  time_to_first_audio_ms: number | null;
  /** Trailing silence the endpointer waited through before committing the turn. */
  endpoint_wait_ms: number | null;
  /**
   * Customer's last word -> first sound out of the speaker. The "voice to
   * voice" clock — the only figure comparable to external voice-kiosk numbers,
   * because unlike turn_total_ms it includes the endpoint wait.
   */
  voice_to_voice_ms: number | null;
  /** Same clock, but to the first audio that carries the actual answer. */
  voice_to_voice_informative_ms: number | null;
  /** Real ASR round-trip on the final chunk (flush-queue join before turn start). */
  final_flush_wait_ms: number | null;
  /**
   * Browser-mic-release turns only: gap between the customer's true last
   * speech frame and the flush sequence starting (button-release reaction
   * time + any trailing buffered frames). Both ends are backend monotonic
   * timestamps — not network/browser-clock overhead. None on the
   * silence-timeout endpoint path, where this gap is ~0 by construction.
   */
  post_speech_gap_ms: number | null;
  /**
   * voice_to_voice_ms minus endpoint_wait_ms — last word -> first sound,
   * WITHOUT the deliberate trailing-silence wait. The number comparable to
   * the lab's "pipeline compute only" clock.
   */
  voice_to_voice_post_endpoint_ms: number | null;
  /** True: the sentence-completeness shortcut fired (didn't wait full silence_timeout). */
  endpoint_shortcut_fired: boolean | null;
}

export interface PipelineAsrSpan {
  ms: number | null;
  device: string;
  chunks?: number;
  /**
   * Continuous-streaming mode only: customer's actual last word -> the
   * moment a transcript covering it landed from the analyzer. The genuine
   * ASR compute latency on the critical path (excludes the trailing-silence
   * wait, which endpoint_wait_ms reports separately). This is the number
   * being tracked against the ~180-220ms target.
   */
  last_word_to_transcript_ms?: number | null;
}

export interface PipelineRetrievalSpan {
  invoked: boolean;
  ms: number | null;
}

export interface PipelineLlmSpan {
  ms: number | null;
  calls: number;
  device: string;
}

export interface PipelineAgentSpan {
  ttft_ms: number | null;
  total_ms: number | null;
  retrieval: PipelineRetrievalSpan;
  llm: PipelineLlmSpan;
}

export interface PipelineTtsSpan {
  ms: number | null;
  device: string;
  segments: number;
  overlapped_with_agent: boolean;
}

export interface PipelineTurnTrace {
  turn_id: string;
  conversation_id: string;
  started_at: string;
  ended_at: string | null;
  wall: PipelineWall;
  asr: PipelineAsrSpan;
  agent: PipelineAgentSpan;
  tts: PipelineTtsSpan;
}

export interface KpiBundle {
  asr: KpiData;
  rag: KpiData;
  tts: KpiData;
  pipeline?: PipelineTurnTrace | null;
  /** Most recent turns, newest last — one entry per request/response. */
  pipelineRecent?: PipelineTurnTrace[];
}

// ── Metrics ─────────────────────────────────────────────────────────────────
// Each series item is [iso_timestamp, value, ...]
export type MetricSeriesItem = [string, ...number[]];

export interface MetricsResponse {
  cpu_utilization?: MetricSeriesItem[];
  gpu_utilization?: MetricSeriesItem[];
  npu_utilization?: MetricSeriesItem[];
  memory?: MetricSeriesItem[];
  [key: string]: unknown;
}

export interface ChartPoint {
  time: string;
  value: number;
}

// ── Voice session UI state ──────────────────────────────────────────────────
export type VoicePhase = 'idle' | 'listening' | 'processing' | 'speaking';

// ── TTS playback queue ──────────────────────────────────────────────────────
export type TtsPlaybackState = 'idle' | 'queued' | 'playing';

// ── Performance Dashboard ───────────────────────────────────────────────────

/** One stage in the AI pipeline */
export type PipelineStageId = 'asr' | 'retrieval' | 'llm' | 'tts';

export interface PipelineStage {
  id: PipelineStageId;
  label: string;
  shortLabel: string;
  device: string;
  model: string;
  latencyMs: number | null;
  /** Is this stage currently active in the ongoing turn? */
  active: boolean;
  color: string;        // Tailwind class prefix, e.g. 'asr'
  iconEmoji: string;
}

/** Derived from KpiBundle for executive display */
export interface ExecutiveKpi {
  id: string;
  label: string;
  value: string;
  unit: string;
  sub: string;
  color: string;  // Tailwind text color class, e.g. 'text-cpu'
  trend?: 'up' | 'down' | 'neutral';
}

/** Latest snapshot of hardware utilization */
export interface HardwareSnapshot {
  cpuPct: number;
  gpuPct: number;
  npuPct: number;
  memPct: number;
}

// ── Identity (biometric auth) ───────────────────────────────────────────────
export interface LoyaltyProfile {
  user_id: string;
  name: string;
  favorites: string[];
  restrictions: string[];
}

export interface ChallengeResponse {
  challenge_id: string;
  prompt_text: string;
}

export interface VerifyRequest {
  challenge_id?: string | null;
  image_base64: string;
  audio_base64: string;
}

export interface VerifyResponse {
  verified: boolean;
  user_id?: string | null;
  profile?: LoyaltyProfile | null;
  face_similarity?: number | null;
  voice_similarity?: number | null;
  fused_score?: number | null;
  reason?: string | null;
}

export interface RegisterRequest {
  user_id: string;
  name: string;
  favorites?: string[];
  restrictions?: string[];
  image_base64?: string | null;
  audio_base64?: string | null;
}

export interface RegisterResponse {
  user_id: string;
  registered: boolean;
  face_faiss_id?: number | null;
  voice_faiss_id?: number | null;
  reason?: string | null;
}

export interface IdentityEnabledResponse {
  enabled: boolean;
}

