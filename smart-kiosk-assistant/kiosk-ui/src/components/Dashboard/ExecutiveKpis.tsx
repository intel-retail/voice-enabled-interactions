/**
 * ExecutiveKpis — large KPI cards visible at all times on the dashboard.
 *
 * Cards:
 *   1. V2V Latency      (customer's last word → first sound out of the speaker)
 *   2. ASR Speed        (last spoken word → transcript ready)
 *   3. LLM Latency      (time to first token / TTFT)
 *   4. TTS Speed        (time to first audio, after LLM TTFT)
 *
 * Source of truth is the per-turn trace at kiosk-core /api/v1/pipeline/latest,
 * routed through the SAME extraction helper as PipelineFlow
 * (utils/turnLatency.ts), so the two panels always agree. They previously
 * each re-derived these numbers independently (this panel summed the raw
 * cumulative trace fields -- asr.ms, agent.llm.ms, tts.ms -- while
 * PipelineFlow used the critical-path numbers instead), which is why the
 * same turn could show different ASR/LLM/TTS figures in each panel.
 *
 * The global registers are still used, but only as a fallback before the first
 * turn has been recorded; that state is labelled so it is never mistaken for
 * a measured turn. V2V has no legacy register to fall back to (it only exists
 * on the per-turn trace), so it simply reads "—" pre-first-turn.
 *
 * Designed for executive demos and trade-show large displays.
 * Cards glow subtly on update (animation: kpi-glow).
 */

import type { KpiBundle } from '../../types';
import { extractLatencies, formatLatency } from '../../utils/turnLatency';

const s = (v: unknown) => (v === null || v === undefined || v === '' ? '—' : String(v));
const tail = (v: unknown) => s(v).split('/').pop() ?? '—';
// Numeric part / unit split of the shared formatLatency string ("148 ms" /
// "1.4 s"), so KpiCard's separate value+unit slots stay pixel-identical to
// the combined string PipelineFlow renders for the same number.
const latencyValue = (v: number | null): string => formatLatency(v).split(' ')[0];
const latencyUnit = (v: number | null): string =>
  v === null ? '' : (formatLatency(v).split(' ')[1] ?? '');

interface KpiCardProps {
  icon: string;
  title: string;
  value: string;
  unit: string;
  sub: string;
  accentCls: string;    // Tailwind border + glow color class
  valueCls: string;     // Tailwind text color for value
  updated: boolean;     // triggers glow animation
}

function KpiCard({ icon, title, value, unit, sub, accentCls, valueCls, updated }: KpiCardProps) {
  return (
    <div
      className={`
        relative flex flex-col rounded-xl border bg-white p-4 transition-all duration-300
        ${accentCls}
        ${updated ? 'animate-kpi-glow' : ''}
      `}
    >
      {/* Top: icon + title */}
      <div className="mb-3 flex items-center gap-2">
        <span className="text-xl leading-none">{icon}</span>
        <span className="text-[11px] font-semibold uppercase tracking-widest text-gray-400">
          {title}
        </span>
      </div>

      {/* Main value */}
      <div className="flex items-baseline gap-1">
        <span
          className={`text-4xl font-bold font-mono leading-none tracking-tight ${valueCls}`}
          style={{ animation: value !== '—' ? 'number-tick 0.25s ease-out' : undefined }}
          key={value}
        >
          {value}
        </span>
        {unit && (
          <span className={`text-base font-semibold ${valueCls} opacity-70`}>{unit}</span>
        )}
      </div>

      {/* Sub-label */}
      <p className="mt-2 text-[11px] leading-snug text-gray-400">{sub}</p>
    </div>
  );
}

interface ExecutiveKpisProps {
  kpis: KpiBundle;
}

export function ExecutiveKpis({ kpis }: ExecutiveKpisProps) {
  // Per-turn trace is authoritative; extractLatencies falls back to the
  // legacy global last_ms registers only before the first turn is recorded.
  const trace = kpis.pipeline ?? null;
  const live = trace !== null;
  const lats = extractLatencies(kpis);

  // Customer's last word -> first sound out of the speaker. Has no legacy
  // global-register fallback (only exists on the per-turn trace).
  const v2vMs = live ? trace.wall.voice_to_voice_ms : null;
  const asrMs = lats.asr;
  const llmMs = lats.llm;
  const ttsMs = lats.tts;
  const retrievalMs = lats.retrievalInvoked ? lats.retrieval : null;

  const llmCalls = lats.llmCalls;
  const ttsSegments = live ? trace.tts.segments : 0;
  const sourceNote = live ? 'measured wall-clock, last turn' : 'awaiting first turn';

  // Build device sub-labels
  const asrDevice = s(kpis.asr?.device).toUpperCase() || '—';
  const llmDevice = s((kpis.rag as Record<string, unknown>)?.llm_device).toUpperCase() || '—';
  const ttsDevice = s(kpis.tts?.device).toUpperCase() || '—';

  const asrModel = tail(kpis.asr?.model);
  const llmModel = tail((kpis.rag as Record<string, unknown>)?.llm_model);
  const ttsModel = tail(kpis.tts?.model);

  return (
    <div className="space-y-2">
      {/* Section header */}
      <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-400">
        Performance KPIs
      </h2>

      {/* Card grid — V2V leads as the customer-facing headline number */}
      <div className="grid grid-cols-2 gap-3">
        {/* V2V Latency */}
        <KpiCard
          icon="🗣️"
          title="V2V Latency"
          value={latencyValue(v2vMs)}
          unit={latencyUnit(v2vMs)}
          sub={`Last word → first sound out · ${sourceNote}`}
          accentCls="border-purple-400/40"
          valueCls="text-purple-700"
          updated={v2vMs !== null}
        />

        {/* ASR Speed */}
        <KpiCard
          icon="🎙"
          title="ASR Speed"
          value={latencyValue(asrMs)}
          unit={latencyUnit(asrMs)}
          sub={`${asrModel} · ${asrDevice}`}
          accentCls="border-asr/40"
          valueCls="text-asr"
          updated={asrMs !== null}
        />

        {/* LLM Generation */}
        <KpiCard
          icon="🧠"
          title="LLM Latency"
          value={latencyValue(llmMs)}
          unit={latencyUnit(llmMs)}
          sub={`${llmModel} · ${llmDevice}${llmCalls > 0 ? ` · ${llmCalls} call${llmCalls > 1 ? 's' : ''}` : ''}`}
          accentCls="border-llm/40"
          valueCls="text-llm"
          updated={llmMs !== null}
        />

        {/* TTS Speed */}
        <KpiCard
          icon="🔊"
          title="TTS Speed"
          value={latencyValue(ttsMs)}
          unit={latencyUnit(ttsMs)}
          sub={`${ttsModel} · ${ttsDevice}${ttsSegments > 0 ? ` · ${ttsSegments} seg` : ''}`}
          accentCls="border-tts/40"
          valueCls="text-tts"
          updated={ttsMs !== null}
        />
      </div>

      {/* Secondary metrics row — Retrieval + docs */}
      <div className="grid grid-cols-2 gap-3">
        <div className="flex items-center gap-3 rounded-lg border border-ret/30 bg-white px-3 py-2">
          <span className="text-lg">🔍</span>
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">
              Retrieval
            </p>
            <p className="font-mono text-lg font-bold text-ret">
              {latencyValue(retrievalMs)}
              <span className="ml-1 text-xs font-normal opacity-70">{latencyUnit(retrievalMs)}</span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 rounded-lg border border-gpu/30 bg-white px-3 py-2">
          <span className="text-lg">📚</span>
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">
              Docs Indexed
            </p>
            <p className="font-mono text-lg font-bold text-gpu">
              {s((kpis.rag as Record<string, unknown>)?.document_count)}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ExecutiveKpis;
