import { useCallback, useEffect, useState } from 'react';

import MenuPanel from '../Order/MenuPanel';
import OrderPanel from '../Order/OrderPanel';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type SubTab = 'menu' | 'cart';
type QueueStatus = 'LOW' | 'MEDIUM' | 'HIGH' | 'unknown';

interface QueueInfo {
  count: number;
  status: QueueStatus;
}

interface QsrPanelProps {
  orderActive: boolean;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const QUEUE_STREAM_URL = '/queue-svc/stream';
const QUEUE_COUNT_URL  = '/queue-svc/api/v1/queue/count';
const QUEUE_POLL_MS    = 2_000;

const STATUS_STYLE: Record<QueueStatus, string> = {
  LOW:     'bg-green-50  border-green-200  text-green-800',
  MEDIUM:  'bg-amber-50  border-amber-200  text-amber-800',
  HIGH:    'bg-red-50    border-red-200    text-red-800',
  unknown: 'bg-gray-50   border-gray-200   text-gray-500',
};

const STATUS_ICON: Record<QueueStatus, string> = {
  LOW: '🟢', MEDIUM: '🟡', HIGH: '🔴', unknown: '⚪',
};

// ---------------------------------------------------------------------------
// useQueueCount — polls the queue-service count endpoint every QUEUE_POLL_MS
// ---------------------------------------------------------------------------

function useQueueCount(
  url: string,
  intervalMs: number,
  onData: (info: QueueInfo) => void,
) {
  useEffect(() => {
    let cancelled = false;

    const poll = async () => {
      try {
        const res = await fetch(url, { signal: AbortSignal.timeout(4000) });
        if (!res.ok || cancelled) return;
        const data = await res.json() as { count: number; status: string };
        onData({ count: data.count ?? 0, status: (data.status as QueueStatus) ?? 'unknown' });
      } catch {
        // queue-service unavailable — banner stays hidden
      }
    };

    void poll();
    const id = window.setInterval(() => { void poll(); }, intervalMs);
    return () => { cancelled = true; window.clearInterval(id); };
  }, [url, intervalMs, onData]);
}

// ---------------------------------------------------------------------------
// QsrPanel — top-level QSR tab
//
// Layout (top → bottom):
//   1. Live MJPEG feed from queue-service (bounding boxes + count overlay)
//   2. Queue status banner  (count + LOW/MEDIUM/HIGH), updates every 2 s
//   3. Peak-hour notice     (MEDIUM/HIGH only, while showing express menu)
//   4. Sub-tab pills        (Menu / Cart)
//   5. MenuPanel + OrderPanel — BOTH always mounted so OrderPanel keeps
//      its polling interval alive; CSS `hidden` class toggles visibility.
// ---------------------------------------------------------------------------

export function QsrPanel({ orderActive }: QsrPanelProps) {
  const [subTab, setSubTab]             = useState<SubTab>('menu');
  const [queueInfo, setQueueInfo]       = useState<QueueInfo | null>(null);
  const [showFullMenu, setShowFullMenu] = useState(false);
  const [streamErr, setStreamErr]       = useState(false);

  const onQueueData = useCallback((info: QueueInfo) => {
    setQueueInfo(info);
  }, []);

  useQueueCount(QUEUE_COUNT_URL, QUEUE_POLL_MS, onQueueData);

  const status  = queueInfo?.status ?? 'unknown';
  const isPeak  = status === 'MEDIUM' || status === 'HIGH';
  const peakOnly = isPeak && !showFullMenu;

  const subTabs: { id: SubTab; label: string; icon: string }[] = [
    { id: 'menu', label: 'Menu', icon: '🍔' },
    { id: 'cart', label: 'Cart', icon: '🛒' },
  ];

  return (
    <div className="space-y-3 p-4">

      {/* 1 ── Live queue feed (MJPEG) ───────────────────────────────────── */}
      <div className="overflow-hidden rounded-lg border border-gray-200 bg-black shadow-sm">
        {streamErr ? (
          <div className="flex items-center justify-center text-xs text-gray-400" style={{ height: '280px' }}>
            📷 Queue feed unavailable
          </div>
        ) : (
          <img
            src={QUEUE_STREAM_URL}
            alt="Live queue feed with person detection"
            className="w-full object-contain"
            style={{ height: '280px' }}
            onError={() => setStreamErr(true)}
          />
        )}
      </div>

      {/* 2 ── Queue status banner ───────────────────────────────────────── */}
      {queueInfo !== null && (
        <div
          className={`flex items-center justify-between rounded-lg border px-3 py-2 text-xs font-medium ${STATUS_STYLE[status]}`}
        >
          <span>
            {STATUS_ICON[status]}&nbsp;Queue:&nbsp;
            <strong>{queueInfo.count}</strong>&nbsp;
            {queueInfo.count === 1 ? 'person' : 'people'}&nbsp;·&nbsp;{status}
          </span>
          {isPeak && (
            <button
              type="button"
              onClick={() => setShowFullMenu((v) => !v)}
              className="ml-2 rounded border border-current px-2 py-0.5 text-[10px] opacity-75 hover:opacity-100"
            >
              {showFullMenu ? '⚡ Peak menu' : '📋 Full menu'}
            </button>
          )}
        </div>
      )}

      {/* 3 ── Peak-hour notice ──────────────────────────────────────────── */}
      {peakOnly && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          ⚡ <strong>Peak hours</strong> — express menu shown (Burgers · Sides · Beverages).
          Tap "Full menu" to see all items.
        </div>
      )}

      {/* 4 ── Sub-tab pills ─────────────────────────────────────────────── */}
      <div className="flex gap-2">
        {subTabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setSubTab(tab.id)}
            className={`
              flex flex-1 items-center justify-center gap-1.5 rounded-lg border py-2 text-xs font-semibold
              transition-colors duration-150
              ${subTab === tab.id
                ? 'border-intel-blue bg-blue-50/60 text-intel-blue'
                : 'border-gray-200 bg-white text-gray-400 hover:text-gray-600'}
            `}
          >
            <span>{tab.icon}</span>
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      {/* 5 ── Panel bodies — BOTH always mounted; CSS toggles visibility ── */}
      <div className={subTab === 'menu' ? '' : 'hidden'}>
        <MenuPanel peakOnly={peakOnly} />
      </div>
      <div className={subTab === 'cart' ? '' : 'hidden'}>
        <OrderPanel active={orderActive} />
      </div>
    </div>
  );
}

export default QsrPanel;
