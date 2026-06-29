import { useCallback, useMemo, useState } from 'react';
import Header from './components/Header/Header';
import Footer from './components/Footer/Footer';
import ChatPane from './components/Chat/ChatPane';
import MicButton from './components/Chat/MicButton';
import AssistantIndicator from './components/Chat/AssistantIndicator';
import { PerformanceDashboard } from './components/Dashboard/PerformanceDashboard';
import useMicDevices from './hooks/useMicDevices';
import { useKpis } from './hooks/useKpis';
import { useVoiceSession } from './hooks/useVoiceSession';
import { useMetrics } from './hooks/useMetrics';
import { toChartPoints } from './api/metricsApi';

export default function App() {
  const { devices, selectedId, setSelectedId, error: micError } = useMicDevices();
  const { kpis, refresh: refreshKpis } = useKpis();
  const { metrics } = useMetrics();
  const [ingestBusy, setIngestBusy] = useState(false);

  const onTurnComplete = useCallback(() => {
    refreshKpis();
  }, [refreshKpis]);

  const {
    phase,
    messages,
    partialUser,
    partialAssistant,
    statusText,
    playbackState,
    start,
    stop,
  } = useVoiceSession({ deviceId: selectedId, enabled: !ingestBusy, onTurnComplete });

  const orderActive = phase === 'listening' || phase === 'processing' || playbackState !== 'idle';

  // Extract latest hardware utilization for the header
  const { cpuPct, gpuPct, npuPct } = useMemo(() => {
    const last = (pts: ReturnType<typeof toChartPoints>) =>
      pts.length > 0 ? pts[pts.length - 1].value : null;
    return {
      cpuPct: last(toChartPoints(metrics.cpu_utilization)),
      gpuPct: last(toChartPoints(metrics.gpu_utilization)),
      npuPct: last(toChartPoints(metrics.npu_utilization)),
    };
  }, [metrics]);

  return (
    <div className="flex flex-col h-full bg-kiosk-pane font-text">
      {/* Enhanced header with live hardware pills */}
      <Header phase={phase} cpuPct={cpuPct} gpuPct={gpuPct} npuPct={npuPct} />

      {/* Main split layout — 65% chat : 35% performance dashboard */}
      <main className="flex-1 overflow-hidden">
        <div
          className="h-full mx-auto px-4 py-4 grid gap-4"
          style={{
            maxWidth: '1600px',
            gridTemplateColumns: '1fr 420px',
          }}
        >
          {/* ── Left 65% — Conversational experience ─────────────────────── */}
          <section className="flex flex-col bg-white rounded-xl border border-kiosk-border overflow-hidden min-h-0 shadow-sm">
            {/* Chat area */}
            <ChatPane
              messages={messages}
              partialUser={partialUser}
              partialAssistant={partialAssistant}
              phase={phase}
            />

            {/* Voice controls bar */}
            <div className="shrink-0 border-t border-kiosk-border bg-kiosk-pane/60 px-6 py-4">
              <div className="flex flex-col items-center gap-3">
                <AssistantIndicator phase={phase} playbackState={playbackState} />
                <MicButton phase={phase} locked={ingestBusy} onStart={start} onStop={stop} />
                <p className="text-xs text-kiosk-textlo text-center min-h-[1rem] max-w-sm">
                  {statusText}
                </p>
              </div>
            </div>
          </section>

          {/* ── Right 35% — Performance Dashboard (always visible) ────────── */}
          <PerformanceDashboard
            kpis={kpis}
            metrics={metrics}
            phase={phase}
            orderActive={orderActive}
            devices={devices}
            selectedDeviceId={selectedId}
            onSelectDevice={setSelectedId}
            micError={micError}
            onIngestStateChange={setIngestBusy}
            onRefreshKpis={refreshKpis}
          />
        </div>
      </main>

      <Footer />
    </div>
  );
}
