import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { fetchCurrentOrder, fetchOrder, fetchUpsell } from '../../api/orderingApi';
import { tuning } from '../../constants';
import type { Order, UpsellSuggestion } from '../../types';

interface OrderPanelProps {
  active: boolean;
}

// Indian Rupees, dropping a trailing .0 for whole values (matches the agent's replies).
const formatCurrency = (value: number | undefined): string => {
  const rounded = Math.round((value ?? 0) * 100) / 100;
  return `₹${Number.isInteger(rounded) ? rounded : rounded.toFixed(2)}`;
};

// Matches the order id the agent speaks (e.g. "ORD-11"), no zero-padding.
const formatOrderId = (orderId: number): string => `ORD-${orderId}`;

// Poll quickly during an active ordering session; fall back to the same cadence
// as the performance dashboard so we don't spam the backend when idle.
const ACTIVE_POLL_MS = 2000;
const IDLE_POLL_MS = tuning.perfRefreshMs; // 10 s

export function OrderPanel({ active }: OrderPanelProps) {
  const [order, setOrder] = useState<Order | null>(null);
  const [suggestions, setSuggestions] = useState<UpsellSuggestion[]>([]);
  const mountedRef = useRef(false);
  // Remember the order currently on screen so we can keep showing the confirmed
  // receipt after the draft query stops returning it.
  const shownOrderRef = useRef<Order | null>(null);

  const applyOrder = useCallback(async (next: Order | null) => {
    if (!mountedRef.current) return;
    shownOrderRef.current = next;
    setOrder(next);

    const productIds =
      next && next.status === 'draft' ? next.items?.map((item) => item.product_id) ?? [] : [];
    if (productIds.length > 0) {
      const nextSuggestions = await fetchUpsell(productIds);
      if (!mountedRef.current) return;
      setSuggestions(nextSuggestions);
    } else {
      setSuggestions([]);
    }
  }, []);

  const loadOrder = useCallback(async () => {
    const draft = await fetchCurrentOrder(tuning.userId);
    if (!mountedRef.current) return;

    if (draft) {
      // A live draft exists — always show it (this also replaces a stale receipt
      // once the customer starts a brand-new order).
      await applyOrder(draft);
      return;
    }

    // No draft. If we were showing one, it was just confirmed — fetch the frozen
    // confirmed order by id and keep it on screen as a receipt instead of blanking.
    const shown = shownOrderRef.current;
    if (shown && shown.status !== 'confirmed') {
      const confirmed = await fetchOrder(shown.order_id);
      if (!mountedRef.current) return;
      if (confirmed) {
        await applyOrder(confirmed);
        return;
      }
    }
    // Already showing a confirmed receipt (or nothing) — leave it untouched.
    if (!shown) {
      await applyOrder(null);
    }
  }, [applyOrder]);

  // Single interval whose cadence adapts to the active state.
  // Replaces the previous two-interval bug that caused overlapping polls.
  useEffect(() => {
    mountedRef.current = true;
    void loadOrder();

    const intervalMs = active ? ACTIVE_POLL_MS : IDLE_POLL_MS;
    const intervalId = window.setInterval(() => {
      void loadOrder();
    }, intervalMs);

    return () => {
      mountedRef.current = false;
      window.clearInterval(intervalId);
    };
  }, [active, loadOrder]);

  const visibleSuggestions = useMemo(() => suggestions.slice(0, 3), [suggestions]);
  const items = order?.items ?? [];
  const isConfirmed = order?.status === 'confirmed';

  return (
    <section className="rounded-lg border border-kiosk-border bg-white p-4">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-intel-dark">
          {isConfirmed ? '✅ Order Confirmed' : '🛒 Current Order'}
        </h2>
        {order?.order_id !== undefined ? (
          <span className="text-xs text-kiosk-textlo">#{formatOrderId(order.order_id)}</span>
        ) : null}
      </div>

      {!order ? (
        <p className="py-3 text-sm text-kiosk-textlo">No active order yet. Start ordering by voice.</p>
      ) : (
        <div className="mt-3">
          <div className="space-y-2">
            {items.map((item) => (
              <div key={item.id} className="flex justify-between gap-3">
                <span className="text-sm text-intel-dark">
                  <span className="text-xs">{item.quantity}×</span> {item.product_name}
                </span>
                <span className="text-sm font-medium">{formatCurrency(item.subtotal)}</span>
              </div>
            ))}
          </div>

          <div className="my-3 border-t border-kiosk-border" />

          <div className="flex items-center justify-between gap-3 text-sm font-bold text-intel-dark">
            <div className="flex items-center gap-2">
              <span>Total</span>
              <span
                className={`rounded-full px-2 py-0.5 text-[10px] ${
                  isConfirmed ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'
                }`}
              >
                {order.status}
              </span>
            </div>
            <span>{formatCurrency(order.total)}</span>
          </div>

          {isConfirmed ? (
            <p className="mt-3 rounded-md bg-green-50 px-2 py-1.5 text-center text-xs text-green-700">
              🎉 Thank you! Your order {formatOrderId(order.order_id)} is confirmed.
            </p>
          ) : visibleSuggestions.length > 0 ? (
            <div>
              <h3 className="mb-1 mt-3 text-xs font-semibold text-kiosk-textmd">✨ You might also like</h3>
              {visibleSuggestions.map((suggestion) => (
                <div
                  key={suggestion.product.product_id}
                  className="mb-1 rounded-md border border-kiosk-border bg-kiosk-asst px-2 py-1 text-xs text-intel-dark"
                >
                  {suggestion.product.name} — {suggestion.reason}
                </div>
              ))}
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}

export default OrderPanel;
