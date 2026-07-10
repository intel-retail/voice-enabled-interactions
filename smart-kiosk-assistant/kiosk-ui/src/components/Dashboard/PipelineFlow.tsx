/**
 * PipelineFlow — visualises the AI inference pipeline as a horizontal node
 * graph with per-stage latency chips and animated flow arrows.
 *
 *  🎤 → [ASR] → [Agent/LLM] → [TTS] → 🔊
 *
 * Latency data comes from the turn trace at kiosk-core /api/v1/pipeline/latest.
 * Wall-clock E2E is measured (not summed), so TTS overlap is handled correctly.
 * Retrieval stage shows "—" when not invoked this turn (ordering turns skip it).
 *
 * Color coding  CPU=Blue  GPU=Green  NPU=Purple
 * Stage colors: ASR=Orange  Retrieval=Yellow  LLM=Cyan  TTS=Pink
 */

import type { KpiBundle, PipelineTurnTrace } from '../../types';
import type { VoicePhase } from '../../types';

interface StageConfig {
  id: string;
  label: string;
  icon: string;
  bg: string;
  border: string;
  textColor: string;
  glowColor: string;
}

const STAGES: StageConfig[] = [
  {
    id: 'asr',
    label: 'ASR',
    icon: '🎙',
    bg: 'bg-asr-light',
    border: 'border-asr',
    textColor: 'text-asr-dark',
    glowColor: 'rgba(234,88,12,0.4)',
  },
  {
    id: 'retrieval',
    label: 'Retrieval',
    icon: '🔍',
    bg: 'bg-ret-light',
    border: 'border-ret',
    textColor: 'text-ret-dark',
    glowColor: 'rgba(202,138,4,0.4)',
  },
  {
    id: 'llm',
    label: 'LLM',
    icon: '🧠',
    bg: 'bg-llm-light',
    border: 'border-llm',
    textColor: 'text-llm-dark',
    glowColor: 'rgba(8,145,178,0.4)',
  },
  {
    id: 'tts',
    label: 'TTS',
    icon: '🔊',
    bg: 'bg-tts-light',
    border: 'border-tts',
    textColor: 'text-tts-dark',
    glowColor: 'rgba(219,39,119,0.4)',
  },
];

interface LatencyMap {
  asr: number | null;
  retrieval: number | null;         // null when not invoked this turn
  llm: number | null;               // agent TTFT — perceived latency
  tts: number | null;
  retrievalInvoked: boolean;
}

function extractLatencies(kpis: KpiBundle): LatencyMap {
  const trace = kpis.pipeline as PipelineTurnTrace | null | undefined;

  if (trace) {
    return {
      asr:              trace.asr?.ms ?? null,
      retrieval:        trace.agent?.retrieval?.invoked ? (trace.agent.retrieval.ms ?? null) : null,
      llm:              trace.agent?.ttft_ms ?? null,
      tts:              trace.tts?.ms ?? null,
      retrievalInvoked: trace.agent?.retrieval?.invoked ?? false,
    };
  }

  // Fallback to legacy last_ms registers when no turn trace is available yet
  const ap = (kpis.asr?.perf ?? {}) as Record<string, unknown>;
  const rp = (kpis.rag?.perf ?? {}) as Record<string, unknown>;
  const retr = (rp.retrieval ?? {}) as Record<string, unknown>;
  const llm  = (rp.llm ?? {}) as Record<string, unknown>;
  const tp = (kpis.tts?.perf ?? {}) as Record<string, unknown>;
  const n = (v: unknown) => (typeof v === 'number' ? v : null);
  return {
    asr:              n(ap.last_ms),
    retrieval:        n(retr.last_ms),
    llm:              n(llm.last_ms),
    tts:              n(tp.last_ms),
    retrievalInvoked: true,   // legacy: always show when present
  };
}

function latencyLabel(ms: number | null, invoked = true): string {
  if (!invoked) return '—';
  if (ms === null) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(1)} s`;
}

function deviceBadge(device: unknown): { label: string; cls: string } | null {
  const d = String(device ?? '').toUpperCase();
  if (d.includes('GPU'))  return { label: 'GPU', cls: 'bg-gpu-light text-gpu-dark border-gpu-muted' };
  if (d.includes('NPU'))  return { label: 'NPU', cls: 'bg-npu-light text-npu-dark border-npu-muted' };
  if (d.includes('CPU'))  return { label: 'CPU', cls: 'bg-cpu-light text-cpu-dark border-cpu-muted' };
  return null;
}

function activeStageFromPhase(phase: VoicePhase): string | null {
  if (phase === 'listening')  return 'asr';
  if (phase === 'processing') return 'llm';
  if (phase === 'speaking')   return 'tts';
  return null;
}

interface PipelineFlowProps {
  kpis: KpiBundle;
  phase: VoicePhase;
}

export function PipelineFlow({ kpis, phase }: PipelineFlowProps) {
  const lats = extractLatencies(kpis);
  const trace = kpis.pipeline as PipelineTurnTrace | null | undefined;

  const latencyByStage: Record<string, number | null> = {
    asr:       lats.asr,
    retrieval: lats.retrieval,
    llm:       lats.llm,
    tts:       lats.tts,
  };

  const invokedByStage: Record<string, boolean> = {
    asr:       true,
    retrieval: lats.retrievalInvoked,
    llm:       true,
    tts:       true,
  };

  const deviceByStage: Record<string, unknown> = {
    asr:       kpis.asr?.device ?? trace?.asr?.device,
    retrieval: (kpis.rag as Record<string, unknown>)?.embedding_device,
    llm:       trace?.agent?.llm?.device ?? (kpis.rag as Record<string, unknown>)?.llm_device,
    tts:       kpis.tts?.device ?? trace?.tts?.device,
  };

  const activeStage = activeStageFromPhase(phase);

  // Use measured wall E2E from turn trace; never sum stages (avoids TTS overlap error)
  const e2eMs = trace?.wall?.turn_total_ms ?? null;
  const ttfaMs = trace?.wall?.time_to_first_audio_ms ?? null;

  return (
    <div className="space-y-3">
      {/* Section header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-400">
          AI Inference Pipeline
        </h2>
        <div className="flex items-center gap-2">
          {ttfaMs !== null && (
            <span className="rounded-full bg-green-50 px-2 py-0.5 text-[10px] font-semibold text-green-700 border border-green-200"
              title="Time to first audio — perceived response latency">
              TTFA {latencyLabel(ttfaMs)}
            </span>
          )}
          {e2eMs !== null && (
            <span className="rounded-full bg-intel-blue/10 px-2.5 py-0.5 text-[11px] font-semibold text-intel-blue"
              title="Measured wall-clock E2E (not summed)">
              E2E {latencyLabel(e2eMs)}
            </span>
          )}
        </div>
      </div>

      {/* Pipeline nodes */}
      <div className="flex items-stretch gap-0">
        {/* Input node */}
        <div className="flex flex-col items-center justify-center">
          <div
            className={`flex h-12 w-12 flex-col items-center justify-center rounded-full border-2 bg-white shadow-sm transition-all duration-300 ${
              phase === 'listening'
                ? 'border-asr animate-stage-pulse shadow-asr/30 shadow-md'
                : 'border-gray-200'
            }`}
          >
            <span className="text-lg">🎤</span>
          </div>
          <span className="mt-1 text-[10px] text-gray-400">Input</span>
        </div>

        {STAGES.map((stage, idx) => {
          const isActive = activeStage === stage.id;
          const latMs = latencyByStage[stage.id];
          const invoked = invokedByStage[stage.id];
          const badge = deviceBadge(deviceByStage[stage.id]);

          return (
            <div key={stage.id} className="flex flex-1 items-stretch">
              {/* Arrow connector */}
              <div className="flex items-center justify-center px-1">
                <svg width="24" height="12" viewBox="0 0 24 12" className="overflow-visible">
                  <line
                    x1="0" y1="6" x2="18" y2="6"
                    stroke={isActive ? '#0071c5' : (!invoked ? '#e5e7eb' : '#cbd5e1')}
                    strokeWidth={isActive ? 2.5 : 1.5}
                    strokeDasharray={isActive ? '4 2' : (!invoked ? '3 3' : undefined)}
                    style={isActive ? { animation: 'dash-flow 0.8s linear infinite' } : undefined}
                  />
                  <polygon
                    points="18,2 24,6 18,10"
                    fill={isActive ? '#0071c5' : (!invoked ? '#e5e7eb' : '#cbd5e1')}
                  />
                </svg>
              </div>

              {/* Stage node */}
              <div
                className={`
                  relative flex flex-1 flex-col items-center justify-between rounded-lg border p-2 transition-all duration-300
                  ${invoked ? stage.bg : 'bg-gray-50'} ${invoked ? stage.border : 'border-gray-200'}
                  ${isActive ? 'animate-stage-pulse shadow-lg' : 'shadow-sm hover:shadow-md'}
                  ${!invoked ? 'opacity-50' : ''}
                `}
                style={isActive ? { boxShadow: `0 0 16px 2px ${stage.glowColor}` } : undefined}
                title={stage.id === 'retrieval' && !invoked ? 'Not invoked this turn (ordering path)' :
                       stage.id === 'llm' ? 'Time to first token (TTFT)' : undefined}
              >
                {/* Device badge top-right */}
                {badge && invoked && (
                  <span
                    className={`absolute -right-1 -top-2 rounded-full border px-1.5 py-0 text-[9px] font-bold ${badge.cls}`}
                  >
                    {badge.label}
                  </span>
                )}

                {/* Icon + label */}
                <div className="flex flex-col items-center gap-0.5">
                  <span className="text-base leading-none">{stage.icon}</span>
                  <span className={`text-[10px] font-semibold ${invoked ? stage.textColor : 'text-gray-400'}`}>
                    {stage.label}
                  </span>
                </div>

                {/* Latency chip */}
                <div
                  className={`mt-1 rounded-full px-1.5 py-0.5 text-[10px] font-mono font-semibold ${invoked ? stage.textColor : 'text-gray-400'} bg-white/70`}
                  key={String(latMs)}
                  style={{ animation: latMs !== null ? 'number-tick 0.25s ease-out' : undefined }}
                >
                  {latencyLabel(latMs, invoked)}
                </div>

                {/* Active indicator dot */}
                {isActive && (
                  <span className="absolute -bottom-1 left-1/2 h-2 w-2 -translate-x-1/2 rounded-full bg-intel-blue shadow-sm" />
                )}
              </div>

              {/* Final arrow after last stage */}
              {idx === STAGES.length - 1 && (
                <div className="flex items-center justify-center px-1">
                  <svg width="24" height="12" viewBox="0 0 24 12">
                    <line x1="0" y1="6" x2="18" y2="6" stroke="#cbd5e1" strokeWidth="1.5" />
                    <polygon points="18,2 24,6 18,10" fill="#cbd5e1" />
                  </svg>
                </div>
              )}
            </div>
          );
        })}

        {/* Output node */}
        <div className="flex flex-col items-center justify-center">
          <div
            className={`flex h-12 w-12 flex-col items-center justify-center rounded-full border-2 bg-white shadow-sm transition-all duration-300 ${
              phase === 'speaking'
                ? 'border-tts animate-stage-pulse shadow-tts/30 shadow-md'
                : 'border-gray-200'
            }`}
          >
            <span className="text-lg">🔊</span>
          </div>
          <span className="mt-1 text-[10px] text-gray-400">Output</span>
        </div>
      </div>

      {/* LLM label clarification when turn trace is available */}
      {trace && (
        <p className="text-[9px] text-gray-400 text-right">
          LLM = time-to-first-token · E2E = measured wall-clock (TTS overlaps LLM)
        </p>
      )}
    </div>
  );
}

export default PipelineFlow;

