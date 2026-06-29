import { endpoints } from '../constants';
import type { Order, UpsellSuggestion } from '../types';

/** Fetch the current draft order for a user, or null if none exists. */
export async function fetchCurrentOrder(userId: string): Promise<Order | null> {
  try {
    const res = await fetch(endpoints.currentOrder(userId), { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return null;
    // Backend returns JSON null (200) when no draft order exists.
    const data: Order | null = await res.json();
    return data ?? null;
  } catch {
    return null;
  }
}

/** Fetch rule-based upsell suggestions for a cart's product IDs. */
export async function fetchUpsell(productIds: string[]): Promise<UpsellSuggestion[]> {
  if (productIds.length === 0) return [];
  try {
    const res = await fetch(endpoints.upsell, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ product_ids: productIds }),
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) return [];
    return await res.json();
  } catch {
    return [];
  }
}
