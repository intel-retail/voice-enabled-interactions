/**
 * turnLatency — single source of truth for turning a kiosk-core pipeline
 * trace into the per-stage latency numbers shown across the dashboard.
 *
 * ExecutiveKpis and PipelineFlow both render "ASR / LLM / TTS / V2V" cards
 * from the same trace and MUST show identical numbers for the same stage.
 * They previously each re-derived these values independently:
 *   - PipelineFlow used the v2v-latency-optimisation vocabulary (critical
 *     path only): asr.last_word_to_transcript_ms, agent.ttft_ms, and a
 *     derived TTS-time-to-first-audio slice.
 *   - ExecutiveKpis used the raw cumulative trace registers instead:
 *     asr.ms (all chunks summed), agent.llm.ms (cumulative model time
 *     across every round-trip), tts.ms (every segment's synth time summed).
 * Same trace, three stages, three different numbers per stage -- exactly
 * the mismatch this module exists to prevent. Always extend/adjust the
 * extraction logic here, never re-derive it locally in a component.
 */

import type { KpiBundle, PipelineTurnTrace } from '../types';

export interface LatencyMap {
  asr: number | null; // genuine ASR compute latency (last word -> transcript ready)
  retrieval: number | null; // null when not invoked this turn
  llm: number | null; // LLM time to first token (TTFT) — on the v2v critical path
  llmCalls: number; // number of LLM round-trips this turn
  agentOverhead: number | null; // agent round-trip minus LLM time (tools + framework)
  tts: number | null; // TTS time to first audio (portion of ttfa after LLM TTFT)
  retrievalInvoked: boolean;
}

export function extractLatencies(kpis: KpiBundle): LatencyMap {
  const trace = kpis.pipeline as PipelineTurnTrace | null | undefined;

  if (trace) {
    const ttft = trace.agent?.ttft_ms ?? null;
    const ttfa = trace.wall?.time_to_first_audio_ms ?? null;
    return {
      // Real ASR compute latency on the critical path (last spoken word ->
      // transcript ready), NOT the "all chunks summed" total -- matches the
      // ~180-220ms target tracked during the v2v-latency work. Falls back to
      // the cumulative figure only when the analyzer didn't report it.
      asr: trace.asr?.last_word_to_transcript_ms ?? trace.asr?.ms ?? null,
      retrieval: trace.agent?.retrieval?.invoked ? (trace.agent.retrieval.ms ?? null) : null,
      // LLM time to first token (agent-start -> first reply token/sentence):
      // the actual customer-felt LLM latency on the v2v path, not the
      // cumulative model time across every round-trip in the turn.
      llm: ttft,
      llmCalls: trace.agent?.llm?.calls ?? 0,
      agentOverhead:
        trace.agent?.llm?.ms != null && ttft != null
          ? Math.max(0, ttft - trace.agent.llm.ms)
          : null,
      // TTS time to first audio: the TTS-only slice of time_to_first_audio_ms,
      // i.e. ttfa minus the LLM TTFT already counted above -- NOT the
      // cumulative synth time for every segment in the reply (that overlaps
      // playback and isn't on the critical path to the first sound out).
      tts: ttfa != null && ttft != null ? Math.max(0, ttfa - ttft) : (trace.tts?.ms ?? null),
      retrievalInvoked: trace.agent?.retrieval?.invoked ?? false,
    };
  }

  // Fallback to legacy last_ms registers when no turn trace is available yet
  const ap = (kpis.asr?.perf ?? {}) as Record<string, unknown>;
  const rp = (kpis.rag?.perf ?? {}) as Record<string, unknown>;
  const retr = (rp.retrieval ?? {}) as Record<string, unknown>;
  const llm = (rp.llm ?? {}) as Record<string, unknown>;
  const tp = (kpis.tts?.perf ?? {}) as Record<string, unknown>;
  const n = (v: unknown) => (typeof v === 'number' ? v : null);
  return {
    asr: n(ap.last_ms),
    retrieval: n(retr.last_ms),
    llm: n(llm.last_ms),
    llmCalls: 0,
    agentOverhead: null,
    tts: n(tp.last_ms),
    retrievalInvoked: true, // legacy: always show when present
  };
}

/**
 * Format a latency in milliseconds the same way everywhere: whole ms below
 * 1000, one decimal place in seconds above -- e.g. "148 ms" / "1.4 s".
 * `invoked` renders "—" for stages not run this turn (e.g. retrieval on an
 * ordering-only turn).
 */
export function formatLatency(ms: number | null, invoked = true): string {
  if (!invoked) return '—';
  if (ms === null) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

/** p95 (not just the single latest turn) is the customer-facing target: a
 * single bad tail turn is what a real customer notices, and a "last turn"
 * or median-style view hides it. */
export function percentile(values: number[], p: number): number | null {
  const vals = values
    .filter((v) => typeof v === 'number' && Number.isFinite(v))
    .sort((a, b) => a - b);
  if (vals.length === 0) return null;
  const idx = Math.min(vals.length - 1, Math.max(0, Math.ceil(p * vals.length) - 1));
  return vals[idx];
}
