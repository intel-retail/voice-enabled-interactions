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
  primary_speaker_id: string | null;
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

export interface KpiBundle {
  asr: KpiData;
  rag: KpiData;
  tts: KpiData;
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
