import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import useMicDevices from '../../hooks/useMicDevices';
import { useVoiceSession } from '../../hooks/useVoiceSession';
import { useQueueStream, type QueueInfo } from '../../hooks/useQueue';
import { fetchMenu, clearCurrentOrder } from '../../api/orderingApi';
import { groupByCategory } from '../../menu/categories';
import { tuning } from '../../constants';
import type { Product } from '../../types';

import Header from '../Header/Header';
import { QueueBar } from './QueueBar';
import { CategoryRail } from './CategoryRail';
import { MenuGrid } from './MenuGrid';
import { CartPanel } from './CartPanel';
import { AskBar } from './AskBar';

/**
 * Customer-facing kiosk screen (KIOSK_UI_MODE=customer).
 *
 * A single view, no tabs: queue-aware menu (left/centre) + live cart (right)
 * + a full-width voice "Ask" button docked to the bottom. Menu and cart are
 * display-only — all ordering happens by voice through AskBar, reusing the
 * exact same session/audio hooks as the operator screen so ASR → agent → TTS
 * behaviour is identical between both screens.
 *
 * Layout target: 26–28" kiosk touchscreen, 1920x1080 landscape (primary).
 * Falls back to a stacked layout with a collapsible cart on narrower/portrait
 * screens.
 */
export function CustomerApp() {
  // This screen has no device picker (no Settings tab), so it deliberately
  // never pins a deviceId constraint on getUserMedia (see useVoiceSession
  // call below, which passes deviceId: ''). Kiosk hardware here can expose
  // several input devices at once (e.g. a webcam mic alongside a Bluetooth
  // headset); forcing devices[0] previously grabbed whichever one the
  // browser happened to enumerate first -- not necessarily the OS's actual
  // default input -- which silently ignored a correctly-connected Bluetooth
  // headset. Leaving the constraint unset lets the browser fall back to the
  // OS-level default input device, which is already routed correctly.
  // useMicDevices() is still invoked for its side effect of priming the mic
  // permission prompt early (see its internal getUserMedia call).
  useMicDevices();

  const [queueInfo, setQueueInfo] = useState<QueueInfo | null>(null);
  const [showFullMenu, setShowFullMenu] = useState(false);
  const onQueueData = useCallback((info: QueueInfo) => setQueueInfo(info), []);
  useQueueStream(onQueueData);

  const [products, setProducts] = useState<Product[] | null>(null);

  // A page load == a new customer / new conversation. The cart is server-side
  // (SQLite), so it survives a refresh unless we explicitly discard it; without
  // this the next customer would inherit the previous one's abandoned items.
  // The cart panel is held back until the reset resolves so a stale cart is
  // never rendered, even for one poll cycle.
  const [cartReady, setCartReady] = useState(false);
  useEffect(() => {
    let cancelled = false;
    void clearCurrentOrder(tuning.userId).finally(() => {
      if (!cancelled) setCartReady(true);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    void fetchMenu().then((data) => {
      if (!cancelled) setProducts(data);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const isPeak = queueInfo?.status === 'MEDIUM' || queueInfo?.status === 'HIGH';
  const peakOnly = isPeak && !showFullMenu;
  const categories = useMemo(
    () => (products ? groupByCategory(products, peakOnly) : []),
    [products, peakOnly],
  );

  // Track which category section is in view so the rail can highlight it,
  // and let the rail scroll a section into view when tapped.
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const sectionsRef = useRef(new Map<string, HTMLElement>());
  const registerSection = useCallback((key: string, node: HTMLElement | null) => {
    if (node) sectionsRef.current.set(key, node);
    else sectionsRef.current.delete(key);
  }, []);
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length > 0) {
          const topMost = visible.reduce((a, b) => (a.boundingClientRect.top < b.boundingClientRect.top ? a : b));
          const key = [...sectionsRef.current.entries()].find(([, n]) => n === topMost.target)?.[0];
          if (key) setActiveKey(key);
        }
      },
      { rootMargin: '-10% 0px -70% 0px', threshold: 0.01 },
    );
    for (const node of sectionsRef.current.values()) observer.observe(node);
    return () => observer.disconnect();
  }, [categories]);

  const handleSelectCategory = useCallback((key: string) => {
    sectionsRef.current.get(key)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, []);

  const onTurnComplete = useCallback(() => {
    // Cart polling picks up the new order state on its own cadence; nothing
    // extra to refresh here (unlike the operator screen there are no KPIs).
  }, []);

  const {
    phase,
    messages,
    partialUser,
    partialAssistant,
    statusText,
    playbackState,
    conversationMode,
    startConversation,
    endConversation,
    interruptSpeaking,
  } = useVoiceSession({ deviceId: '', enabled: true, onTurnComplete });

  const cartActive = phase === 'listening' || phase === 'processing' || playbackState !== 'idle';

  return (
    <div className="flex h-full flex-col bg-gray-50">
      <Header queueInfo={queueInfo} />
      <QueueBar
        isPeak={isPeak}
        showFullMenu={showFullMenu}
        onToggleFullMenu={() => setShowFullMenu((v) => !v)}
      />

      <main className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* Category rail: horizontal strip on portrait/narrow, vertical on landscape */}
        <div className="shrink-0 overflow-x-auto border-b border-gray-200 bg-white p-3 lg:w-48 lg:overflow-y-auto lg:border-b-0 lg:border-r lg:p-4">
          <CategoryRail categories={categories} activeKey={activeKey} onSelect={handleSelectCategory} />
        </div>

        {/* Menu */}
        <div className="min-h-0 flex-1 overflow-y-auto">
          <MenuGrid categories={categories} registerSection={registerSection} />
        </div>

        {/* Cart: docked right column on landscape, full-width collapsible-height panel below on narrow screens */}
        <div className="h-64 shrink-0 border-t border-gray-200 lg:h-auto lg:w-80 lg:border-l lg:border-t-0 xl:w-96">
          {cartReady && <CartPanel active={cartActive} />}
        </div>
      </main>

      <AskBar
        phase={phase}
        playbackState={playbackState}
        statusText={statusText}
        partialUser={partialUser}
        partialAssistant={partialAssistant}
        messages={messages}
        conversationMode={conversationMode}
        onStart={startConversation}
        onStop={endConversation}
        onInterrupt={interruptSpeaking}
      />
    </div>
  );
}

export default CustomerApp;
