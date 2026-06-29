/**
 * PipelineFlow — visualises the AI inference pipeline as a horizontal node
 * graph with per-stage latency chips and animated flow arrows.
 *
 *  🎤 → [ASR] → [Retrieval] → [LLM] → [TTS] → 🔊
 *
 * Color coding  CPU=Blue  GPU=Green  NPU=Purple
 * Stage colors: ASR=Orange  Retrieval=Yellow  LLM=Cyan  TTS=Pink
 */

import type { KpiBundle } from '../../types';
import type { VoicePhase } from '../../types';

interface StageConfig {
  id: string;
  label: string;
  icon: string;
  bg: string;         // Tailwind bg class
  border: string;     // Tailwind border class
  textColor: string;  // Tailwind text class
  glowColor: string;  // CSS box-shadow color string
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
  asrMs: number | null;
  retrievalMs: number | null;
  llmMs: number | null;
  ttsMs: number | null;
}

function extractLatencies(kpis: KpiBundle): LatencyMap {
  const ap = (kpis.asr?.perf ?? {}) as Record<string, unknown>;
  const rp = (kpis.rag?.perf ?? {}) as Record<string, unknown>;
  const retr = (rp.retrieval ?? {}) as Record<string, unknown>;
  const llm = (rp.llm ?? {}) as Record<string, unknown>;
  const tp = (kpis.tts?.perf ?? {}) as Record<string, unknown>;

  const n = (v: unknown) => (typeof v === 'number' ? v : null);
  return {
    asrMs:       n(ap.last_ms),
    retrievalMs: n(retr.last_ms),
    llmMs:       n(llm.last_ms),
    ttsMs:       n(tp.last_ms),
  };
}

function latencyLabel(ms: number | null): string {
  if (ms === null) return '—';
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
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
  if (phase === 'processing') return 'retrieval'; // approximate: cycles through
  if (phase === 'speaking')   return 'tts';
  return null;
}

interface PipelineFlowProps {
  kpis: KpiBundle;
  phase: VoicePhase;
}

export function PipelineFlow({ kpis, phase }: PipelineFlowProps) {
  const lats = extractLatencies(kpis);
  const latencyByStage: Record<string, number | null> = {
    asr:       lats.asrMs,
    retrieval: lats.retrievalMs,
    llm:       lats.llmMs,
    tts:       lats.ttsMs,
  };

  const deviceByStage: Record<string, unknown> = {
    asr:       kpis.asr?.device,
    retrieval: (kpis.rag as Record<string, unknown>)?.embedding_device,
    llm:       (kpis.rag as Record<string, unknown>)?.llm_device,
    tts:       kpis.tts?.device,
  };

  const activeStage = activeStageFromPhase(phase);

  // E2E latency: sum of all known stages
  const allLats = [lats.asrMs, lats.retrievalMs, lats.llmMs, lats.ttsMs].filter(
    (v): v is number => v !== null,
  );
  const e2eMs = allLats.length > 0 ? allLats.reduce((a, b) => a + b, 0) : null;

  return (
    <div className="space-y-3">
      {/* Section header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-400">
          AI Inference Pipeline
        </h2>
        {e2eMs !== null && (
          <span className="rounded-full bg-intel-blue/10 px-2.5 py-0.5 text-[11px] font-semibold text-intel-blue">
            E2E {latencyLabel(e2eMs)}
          </span>
        )}
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
          const badge = deviceBadge(deviceByStage[stage.id]);

          return (
            <div key={stage.id} className="flex flex-1 items-stretch">
              {/* Arrow connector */}
              <div className="flex items-center justify-center px-1">
                <svg width="24" height="12" viewBox="0 0 24 12" className="overflow-visible">
                  <line
                    x1="0" y1="6" x2="18" y2="6"
                    stroke={isActive ? '#0071c5' : '#cbd5e1'}
                    strokeWidth={isActive ? 2.5 : 1.5}
                    strokeDasharray={isActive ? '4 2' : undefined}
                    style={isActive ? { animation: 'dash-flow 0.8s linear infinite' } : undefined}
                  />
                  <polygon
                    points="18,2 24,6 18,10"
                    fill={isActive ? '#0071c5' : '#cbd5e1'}
                  />
                </svg>
              </div>

              {/* Stage node */}
              <div
                className={`
                  relative flex flex-1 flex-col items-center justify-between rounded-lg border p-2 transition-all duration-300
                  ${stage.bg} ${stage.border}
                  ${isActive ? 'animate-stage-pulse shadow-lg' : 'shadow-sm hover:shadow-md'}
                `}
                style={isActive ? { boxShadow: `0 0 16px 2px ${stage.glowColor}` } : undefined}
              >
                {/* Device badge top-right */}
                {badge && (
                  <span
                    className={`absolute -right-1 -top-2 rounded-full border px-1.5 py-0 text-[9px] font-bold ${badge.cls}`}
                  >
                    {badge.label}
                  </span>
                )}

                {/* Icon + label */}
                <div className="flex flex-col items-center gap-0.5">
                  <span className="text-base leading-none">{stage.icon}</span>
                  <span className={`text-[10px] font-semibold ${stage.textColor}`}>
                    {stage.label}
                  </span>
                </div>

                {/* Latency chip */}
                <div
                  className={`mt-1 rounded-full px-1.5 py-0.5 text-[10px] font-mono font-semibold ${stage.textColor} bg-white/70`}
                  key={String(latMs)}
                  style={{ animation: latMs !== null ? 'number-tick 0.25s ease-out' : undefined }}
                >
                  {latencyLabel(latMs)}
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
    </div>
  );
}

export default PipelineFlow;
