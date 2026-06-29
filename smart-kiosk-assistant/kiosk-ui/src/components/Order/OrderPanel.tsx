import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { fetchCurrentOrder, fetchUpsell } from '../../api/orderingApi';
import { tuning } from '../../constants';
import type { Order, UpsellSuggestion } from '../../types';

interface OrderPanelProps {
  active: boolean;
}

const formatCurrency = (value: number | undefined): string => `$${(value ?? 0).toFixed(2)}`;

const formatOrderId = (orderId: number): string => `ORD-${String(orderId).padStart(5, '0')}`;

export function OrderPanel({ active }: OrderPanelProps) {
  const [order, setOrder] = useState<Order | null>(null);
  const [suggestions, setSuggestions] = useState<UpsellSuggestion[]>([]);
  const mountedRef = useRef(false);

  const loadOrder = useCallback(async () => {
    const nextOrder = await fetchCurrentOrder(tuning.userId);
    if (!mountedRef.current) return;

    setOrder(nextOrder);

    const productIds = nextOrder?.items?.map((item) => item.product_id) ?? [];
    if (nextOrder && productIds.length > 0) {
      const nextSuggestions = await fetchUpsell(productIds);
      if (!mountedRef.current) return;
      setSuggestions(nextSuggestions);
    } else {
      setSuggestions([]);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void loadOrder();

    const intervalId = window.setInterval(() => {
      void loadOrder();
    }, 3000);

    return () => {
      mountedRef.current = false;
      window.clearInterval(intervalId);
    };
  }, [loadOrder]);

  useEffect(() => {
    if (!active) return undefined;

    void loadOrder();
    const intervalId = window.setInterval(() => {
      void loadOrder();
    }, tuning.pollIntervalMs);

    return () => window.clearInterval(intervalId);
  }, [active, loadOrder]);

  const visibleSuggestions = useMemo(() => suggestions.slice(0, 3), [suggestions]);
  const items = order?.items ?? [];

  return (
    <section className="rounded-lg border border-kiosk-border bg-white p-4">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-intel-dark">🛒 Current Order</h2>
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
                  order.status === 'confirmed'
                    ? 'bg-green-100 text-green-700'
                    : 'bg-amber-100 text-amber-700'
                }`}
              >
                {order.status}
              </span>
            </div>
            <span>{formatCurrency(order.total)}</span>
          </div>

          {order.status === 'draft' && visibleSuggestions.length > 0 ? (
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
