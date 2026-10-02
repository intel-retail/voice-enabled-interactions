/**
 * PipelineFlow — visualises the AI inference pipeline as a horizontal node
 * graph with per-stage latency chips and animated flow arrows.
 *
 *  🎤 → [ASR] → [Agent/LLM] → [TTS] → 🔊
 *
 * Latency data comes from the turn trace at kiosk-core /api/v1/pipeline/latest.
 * Retrieval stage shows "—" when not invoked this turn (ordering turns skip it).
 * The E2E (wall-clock, capture → last audio) chip is retired as a headline
 * metric — V2V / V2V p95 / Processing are the surfaced customer-facing clocks.
 *
 * Per-stage chip values reflect the v2v-latency-optimisation work's own
 * vocabulary (see benchmark-vocabolary.txt / v2v_scripted_conversation_benchmark.py):
 *   ASR = real compute latency, last spoken word → transcript ready
 *         (asr.last_word_to_transcript_ms), NOT the "all chunks summed" total.
 *   LLM = time to first token (TTFT), agent-start → first reply token/sentence
 *         (agent.ttft_ms), NOT cumulative model time across every round-trip.
 *   TTS = time to first audio, i.e. the TTS-only slice of
 *         wall.time_to_first_audio_ms remaining after the LLM TTFT above.
 *
 * Color coding  CPU=Blue  GPU=Green  NPU=Purple
 * Stage colors: ASR=Orange  Retrieval=Yellow  LLM=Cyan  TTS=Pink
 */

import type { KpiBundle, PipelineTurnTrace } from '../../types';
import type { VoicePhase } from '../../types';
import { extractLatencies, formatLatency as latencyLabel, percentile } from '../../utils/turnLatency';

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

  // Shared-vocabulary "Processing latency" (turn-end decision -> first sound
  // out of the speaker). Deliberately NOT time_to_first_audio_ms: the two
  // differ whenever work starts speculatively during the endpoint's
  // trailing-silence wait (e.g. shortcut turns: ttfa ~195ms vs processing
  // latency ~3ms, because audio was already rendered before the turn-end
  // decision fired) -- see kiosk_core/pipeline_latency.py WallTimes docstring
  // and benchmark-vocabolary.txt. "Time to first audio" is retired as a
  // headline KPI name; it's still available in the per-turn table below for
  // diagnostics, just not surfaced here as a top-level chip.
  const processingMs = trace?.wall?.voice_to_voice_post_endpoint_ms ?? null;
  // Voice-to-voice is the customer-felt clock: last word spoken -> first sound
  // out of the speaker. It is NOT derivable from turn_total_ms (E2E, retired
  // as a headline chip), because turn_total_ms starts at the endpoint
  // decision and so excludes the trailing-silence wait.
  const v2vMs = trace?.wall?.voice_to_voice_ms ?? null;
  const v2vInformativeMs = trace?.wall?.voice_to_voice_informative_ms ?? null;
  // The non-compute chunk of v2v that Processing (above) deliberately
  // excludes: endpoint_wait_ms (trailing-silence dwell) + post_speech_gap_ms
  // (mic-release reaction time / trailing buffered audio) + final_flush_wait_ms
  // (draining the ASR flush queue). This is real, customer-felt wait time —
  // without a chip for it, ASR+LLM+TTS's stage numbers never add up to V2V
  // and look like a bug. Derived as v2v - processing rather than summed
  // directly so it's never inconsistent with the two chips either side of it.
  const endpointWaitMs =
    v2vMs !== null && processingMs !== null ? Math.max(0, v2vMs - processingMs) : null;

  // One row per request/response, newest first.
  const recentTurns = (kpis.pipelineRecent ?? [])
    .filter((t) => t?.wall?.voice_to_voice_ms != null)
    .slice()
    .reverse();

  // Rolling p95 over the recent-turns window -- the customer-facing target
  // metric (a single bad tail turn is what damages a real interaction;
  // showing only the latest turn's V2V, or a mean/median across the window,
  // would hide it).
  const v2vP95Ms = percentile(
    recentTurns.map((t) => t?.wall?.voice_to_voice_ms as number).filter((v) => v != null),
    0.95,
  );

  return (
    <div className="space-y-3">
      {/* Section header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-400">
          AI Inference Pipeline
        </h2>
        <div className="flex items-center gap-2">
          {v2vMs !== null && (
            <span className="rounded-full bg-purple-50 px-2.5 py-0.5 text-[11px] font-semibold text-purple-700 border border-purple-200"
              title={
                'Voice to voice — customer\u2019s last word to first sound out of the speaker' +
                (v2vInformativeMs !== null
                  ? ` (to first informative audio: ${latencyLabel(v2vInformativeMs)})`
                  : '')
              }>
              V2V {latencyLabel(v2vMs)}
            </span>
          )}
          {v2vP95Ms !== null && (
            <span className="rounded-full bg-purple-100 px-2 py-0.5 text-[10px] font-semibold text-purple-800 border border-purple-300"
              title={`V2V p95 over the last ${recentTurns.length} turns — the customer-facing target metric (median hides the bad tail)`}>
              V2V p95 {latencyLabel(v2vP95Ms)}
            </span>
          )}
          {processingMs !== null && (
            <span className="rounded-full bg-green-50 px-2 py-0.5 text-[10px] font-semibold text-green-700 border border-green-200"
              title="Processing latency — turn-end decision to first sound out of the speaker (shared cross-team vocabulary term; voice_to_voice = endpointing delay + processing latency)">
              Processing {latencyLabel(processingMs)}
            </span>
          )}
          {endpointWaitMs !== null && (
            <span className="rounded-full bg-amber-50 px-2 py-0.5 text-[10px] font-semibold text-amber-700 border border-amber-200"
              title="Endpoint/mic wait — trailing-silence dwell + mic-release reaction time + ASR flush-queue drain, all BEFORE processing starts. Real customer-felt wait, not pipeline compute; V2V = this + Processing.">
              Endpoint wait {latencyLabel(endpointWaitMs)}
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
                       stage.id === 'asr'
                         ? 'ASR latency — last spoken word to transcript ready (excludes the endpoint silence wait)'
                       : stage.id === 'llm'
                         ? 'LLM time to first token (TTFT) — agent-start to first reply token/sentence'
                           + (lats.llmCalls > 0
                               ? ` · cumulative model time across ${lats.llmCalls} round-trip(s)`
                                 + (lats.agentOverhead != null
                                     ? ` · +${Math.round(lats.agentOverhead)} ms agent/tool overhead`
                                     : '')
                               : '')
                       : stage.id === 'tts'
                         ? 'TTS time to first audio — first segment synth + WAV write, after the LLM TTFT'
                         : undefined}
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

      {/* Stage-metric clarification when turn trace is available */}
      {trace && (
        <p className="text-[9px] text-gray-400 text-right">
          ASR = last word → transcript ready · LLM = time to first token (TTFT)
          {lats.llmCalls > 0
            ? ` (${lats.llmCalls} model call${lats.llmCalls > 1 ? 's' : ''}`
              + (lats.agentOverhead != null ? `, +${Math.round(lats.agentOverhead)} ms agent/tool overhead)` : ')')
            : ''}
          {' · TTS = time to first audio (TTS overlaps LLM)'}
          {' · V2V = Endpoint wait + Processing (ASR/LLM/TTS chips are compute-only sub-components of Processing)'}
        </p>
      )}

    </div>
  );
}

export default PipelineFlow;

