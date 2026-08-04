/**
 * ExecutiveKpis — four large KPI cards visible at all times on the dashboard.
 *
 * Cards:
 *   1. E2E Latency      (measured wall-clock round-trip for the last turn)
 *   2. ASR Speed        (speech recognition)
 *   3. LLM Latency      (cumulative model time for the turn)
 *   4. TTS Speed        (speech synthesis)
 *
 * Source of truth is the per-turn trace at kiosk-core /api/v1/pipeline/latest —
 * the SAME source as PipelineFlow, so the two panels always agree. Previously
 * this panel summed each service's global `last_ms` register, which is
 * last-call-wins, spans different turns, omitted agent/tool overhead, and
 * therefore produced a third, lower "E2E" number than the pipeline panel.
 *
 * The global registers are still used, but only as a fallback before the first
 * turn has been recorded; that state is labelled so it is never mistaken for
 * a measured turn.
 *
 * Designed for executive demos and trade-show large displays.
 * Cards glow subtly on update (animation: kpi-glow).
 */

import type { KpiBundle } from '../../types';

const s = (v: unknown) => (v === null || v === undefined || v === '' ? '—' : String(v));
const tail = (v: unknown) => s(v).split('/').pop() ?? '—';
const ms = (v: unknown): string =>
  typeof v === 'number' ? (v < 1000 ? `${Math.round(v)}` : `${(v / 1000).toFixed(2)}`) : '—';
const msUnit = (v: unknown): string =>
  typeof v === 'number' ? (v < 1000 ? 'ms' : 's') : '';

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
  const ap = (kpis.asr?.perf ?? {}) as Record<string, unknown>;
  const rp = (kpis.rag?.perf ?? {}) as Record<string, unknown>;
  const retr = (rp.retrieval ?? {}) as Record<string, unknown>;
  const llm = (rp.llm ?? {}) as Record<string, unknown>;
  const tp = (kpis.tts?.perf ?? {}) as Record<string, unknown>;

  const num = (v: unknown): number | null => (typeof v === 'number' ? v : null);

  // Per-turn trace is authoritative; global registers are only a pre-first-turn
  // fallback so the cards are not blank on a freshly started stack.
  const trace = kpis.pipeline ?? null;
  const live = trace !== null;

  const e2eMs = live ? trace.wall.turn_total_ms : null;
  const asrMs = live ? trace.asr.ms : num(ap.last_ms);
  const llmMs = live ? trace.agent.llm.ms : num(llm.last_ms);
  const ttsMs = live ? trace.tts.ms : num(tp.last_ms);
  const retrievalMs = live
    ? trace.agent.retrieval.invoked
      ? trace.agent.retrieval.ms
      : null
    : num(retr.last_ms);

  const llmCalls = live ? trace.agent.llm.calls : 0;
  const ttsSegments = live ? trace.tts.segments : 0;
  const sourceNote = live ? 'measured wall-clock, last turn' : 'awaiting first turn';

  // Build device sub-labels
  const asrDevice = s(kpis.asr?.device).toUpperCase() || '—';
  const llmDevice = s((kpis.rag as Record<string, unknown>)?.llm_device).toUpperCase() || '—';
  const ttsDevice = s(kpis.tts?.device).toUpperCase() || '—';

  const asrModel = tail(kpis.asr?.model);
  const llmModel = tail((kpis.rag as Record<string, unknown>)?.llm_model);
  const ttsModel = tail(kpis.tts?.model);

  const hasData = e2eMs !== null || asrMs !== null || llmMs !== null || ttsMs !== null;

  return (
    <div className="space-y-2">
      {/* Section header */}
      <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-400">
        Performance KPIs
      </h2>

      {/* 2 × 2 card grid */}
      <div className="grid grid-cols-2 gap-3">
        {/* E2E Latency */}
        <KpiCard
          icon="⚡"
          title="E2E Latency"
          value={ms(e2eMs)}
          unit={msUnit(e2eMs)}
          sub={`Capture → last audio · ${sourceNote}`}
          accentCls="border-intel-blue/40"
          valueCls="text-intel-blue"
          updated={hasData}
        />

        {/* ASR Speed */}
        <KpiCard
          icon="🎙"
          title="ASR Speed"
          value={ms(asrMs)}
          unit={msUnit(asrMs)}
          sub={`${asrModel} · ${asrDevice}`}
          accentCls="border-asr/40"
          valueCls="text-asr"
          updated={asrMs !== null}
        />

        {/* LLM Generation */}
        <KpiCard
          icon="🧠"
          title="LLM Latency"
          value={ms(llmMs)}
          unit={msUnit(llmMs)}
          sub={`${llmModel} · ${llmDevice}${llmCalls > 0 ? ` · ${llmCalls} call${llmCalls > 1 ? 's' : ''}` : ''}`}
          accentCls="border-llm/40"
          valueCls="text-llm"
          updated={llmMs !== null}
        />

        {/* TTS Speed */}
        <KpiCard
          icon="🔊"
          title="TTS Speed"
          value={ms(ttsMs)}
          unit={msUnit(ttsMs)}
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
              {ms(retrievalMs)}
              <span className="ml-1 text-xs font-normal opacity-70">{msUnit(retrievalMs)}</span>
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
