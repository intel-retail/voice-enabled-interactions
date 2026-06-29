import { useCallback, useState } from 'react';
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

  return (
    <div className="flex flex-col h-full bg-gray-100 font-text">
      <Header />

      {/*
        Responsive layout:
          • Mobile/tablet (<lg): single column — chat on top, dashboard below
          • Desktop (lg+):       two columns — 60% chat | 40% dashboard, both fill height
        Padding is kept tight (p-3) so panels reach near screen edges.
        No maxWidth / mx-auto so the layout fills the full viewport width.
      */}
      <main className="flex-1 min-h-0">
        <div
          className="h-full p-3 flex flex-col gap-3 lg:grid lg:gap-3"
          style={{
            gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)',
            gridTemplateRows: '1fr',
          }}
        >
          {/* ── Chat pane ─────────────────────────────────────────────────── */}
          <section
            className="flex flex-col bg-white rounded-xl border border-gray-200 overflow-hidden shadow-sm
                       min-h-[420px] lg:min-h-0"
          >
            <ChatPane
              messages={messages}
              partialUser={partialUser}
              partialAssistant={partialAssistant}
              phase={phase}
            />
            <div className="shrink-0 border-t border-gray-200 bg-gray-50/80 px-4 sm:px-6 py-3 sm:py-4">
              <div className="flex flex-col items-center gap-2 sm:gap-3">
                <AssistantIndicator phase={phase} playbackState={playbackState} />
                <MicButton phase={phase} locked={ingestBusy} onStart={start} onStop={stop} />
                <p className="text-xs text-kiosk-textlo text-center min-h-[1rem] max-w-sm">
                  {statusText}
                </p>
              </div>
            </div>
          </section>

          {/* ── Performance & Settings — direct grid item, fills its column ── */}
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
