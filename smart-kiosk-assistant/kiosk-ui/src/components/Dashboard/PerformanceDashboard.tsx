/**
 * PerformanceDashboard — always-visible right-side panel (35% of layout).
 *
 * Tab 1 — PERFORMANCE (default):
 *   ┌─ AI Inference Pipeline ─────────────────────┐
 *   │  🎤 → [ASR] → [Retrieval] → [LLM] → [TTS] → 🔊 │
 *   └─────────────────────────────────────────────┘
 *   ┌─ Performance KPIs ──────────────────────────┐
 *   │  ⚡ E2E  |  🎙 ASR  |  🧠 LLM  |  🔊 TTS  │
 *   └─────────────────────────────────────────────┘
 *   ┌─ Hardware Utilization (live) ───────────────┐
 *   │  CPU ██░░  GPU ██░░  NPU ██░░               │
 *   │  [CPU area chart]  [GPU area chart]          │
 *   │  [NPU area chart]  [MEM area chart]          │
 *   └─────────────────────────────────────────────┘
 *
 * Tab 2 — SETTINGS (secondary):
 *   Device Settings, Knowledge Base, Current Order
 */

import { useState } from 'react';

import type { KpiBundle, MetricsResponse, VoicePhase } from '../../types';
import { PipelineFlow } from './PipelineFlow';
import { ExecutiveKpis } from './ExecutiveKpis';
import { HardwareCharts } from './HardwareCharts';
import DeviceSettings from '../Panels/DeviceSettings';
import KnowledgeBase from '../Panels/KnowledgeBase';
import OrderPanel from '../Order/OrderPanel';

type Tab = 'performance' | 'settings';

interface PerformanceDashboardProps {
  kpis: KpiBundle;
  metrics: MetricsResponse;
  phase: VoicePhase;
  orderActive: boolean;
  // DeviceSettings props
  devices: MediaDeviceInfo[];
  selectedDeviceId: string;
  onSelectDevice: (id: string) => void;
  micError: string | null;
  // KB ingest callback
  onIngestStateChange: (busy: boolean) => void;
  onRefreshKpis: () => void;
}

export function PerformanceDashboard({
  kpis,
  metrics,
  phase,
  orderActive,
  devices,
  selectedDeviceId,
  onSelectDevice,
  micError,
  onIngestStateChange,
  onRefreshKpis,
}: PerformanceDashboardProps) {
  const [activeTab, setActiveTab] = useState<Tab>('performance');

  return (
    <aside className="flex h-full flex-col overflow-hidden rounded-xl border border-gray-200 bg-gray-50 shadow-sm">
      {/* ── Tab bar ─────────────────────────────────────────────────────────── */}
      <div className="flex shrink-0 border-b border-gray-200 bg-white">
        {(
          [
            { id: 'performance' as Tab, label: 'Performance', icon: '📊' },
            { id: 'settings'    as Tab, label: 'Settings',    icon: '⚙️' },
          ] as { id: Tab; label: string; icon: string }[]
        ).map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={`
              flex flex-1 items-center justify-center gap-2 border-b-2 py-3 text-xs font-semibold
              uppercase tracking-widest transition-colors duration-150
              ${
                activeTab === tab.id
                  ? 'border-intel-blue text-intel-blue bg-blue-50/50'
                  : 'border-transparent text-gray-400 hover:text-gray-600 hover:bg-gray-50'
              }
            `}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* ── Scrollable body ─────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'performance' && (
          <div className="space-y-5 p-4">
            {/* AI Pipeline */}
            <PipelineFlow kpis={kpis} phase={phase} />

            <div className="h-px bg-gray-200" />

            {/* Executive KPIs */}
            <ExecutiveKpis kpis={kpis} />

            <div className="flex justify-end">
              <button
                type="button"
                onClick={onRefreshKpis}
                className="flex items-center gap-1.5 rounded-md border border-gray-200 bg-white px-2.5 py-1.5 text-[11px] text-gray-500 transition-colors hover:border-intel-blue/50 hover:text-intel-blue"
              >
                <svg className="h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8" />
                  <path d="M21 3v5h-5" />
                  <path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16" />
                  <path d="M8 16H3v5" />
                </svg>
                Refresh KPIs
              </button>
            </div>

            <div className="h-px bg-gray-200" />

            {/* Hardware Charts */}
            <HardwareCharts metrics={metrics} />
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="space-y-4 p-4">
            {/* Current Order */}
            <div className="overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm">
              <div className="border-b border-gray-100 bg-gray-50 px-3 py-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
                  🛒 Current Order
                </span>
              </div>
              <div className="p-0">
                <OrderPanel active={orderActive} />
              </div>
            </div>

            {/* Device Settings */}
            <div className="overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm">
              <div className="border-b border-gray-100 bg-gray-50 px-3 py-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
                  🎙 Audio Device
                </span>
              </div>
              <div className="p-3">
                <DeviceSettings
                  devices={devices}
                  selectedId={selectedDeviceId}
                  onSelect={onSelectDevice}
                  error={micError}
                />
              </div>
            </div>

            {/* Knowledge Base */}
            <div className="overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm">
              <div className="border-b border-gray-100 bg-gray-50 px-3 py-2">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
                  📚 Knowledge Base
                </span>
              </div>
              <div className="p-3">
                <KnowledgeBase onIngestStateChange={onIngestStateChange} />
              </div>
            </div>

            {/* Model details */}
            <ModelDetails kpis={kpis} />
          </div>
        )}
      </div>
    </aside>
  );
}

// ── Compact model details in settings tab ───────────────────────────────────
function ModelDetails({ kpis }: { kpis: KpiBundle }) {
  const s = (v: unknown) => (v === null || v === undefined || v === '' ? '—' : String(v));
  const tail = (v: unknown) => s(v).split('/').pop() ?? '—';

  const rows: [string, string, string][] = [
    ['🎙 ASR',       tail(kpis.asr?.model),                                    s(kpis.asr?.device).toUpperCase()],
    ['🔍 Embedding', tail((kpis.rag as Record<string,unknown>)?.embedding_model), s((kpis.rag as Record<string,unknown>)?.embedding_device).toUpperCase()],
    ['🧠 LLM',       tail((kpis.rag as Record<string,unknown>)?.llm_model),    s((kpis.rag as Record<string,unknown>)?.llm_device).toUpperCase()],
    ['🔊 TTS',       tail(kpis.tts?.model),                                    s(kpis.tts?.device).toUpperCase()],
  ];

  return (
    <div className="overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm">
      <div className="border-b border-gray-100 bg-gray-50 px-3 py-2">
        <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
          ⚙️ Model Configuration
        </span>
      </div>
      <div className="divide-y divide-gray-100">
        {rows.map(([stage, model, device]) => (
          <div key={stage} className="flex items-center justify-between gap-2 bg-white px-3 py-2">
            <span className="shrink-0 text-xs text-gray-500">{stage}</span>
            <span className="min-w-0 flex-1 truncate text-center text-[11px] font-medium text-gray-700">
              {model}
            </span>
            <span className="shrink-0 rounded border border-gray-200 bg-gray-50 px-1.5 py-0.5 text-[9px] font-bold text-gray-500">
              {device}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default PerformanceDashboard;
