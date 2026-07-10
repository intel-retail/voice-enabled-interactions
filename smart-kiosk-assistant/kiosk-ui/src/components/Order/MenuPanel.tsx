import { useEffect, useMemo, useRef, useState } from 'react';

import { fetchMenu } from '../../api/orderingApi';
import type { Product } from '../../types';

/** Format a price in Indian Rupees, dropping trailing .0 for whole values. */
const formatPrice = (value: number): string => {
  const rounded = Math.round(value * 100) / 100;
  return `₹${Number.isInteger(rounded) ? rounded : rounded.toFixed(2)}`;
};

// Display order + icon for each catalogue category. Categories not listed here
// still render (alphabetically, with a default icon) so the menu never hides items.
const CATEGORY_META: { key: string; label: string; icon: string }[] = [
  { key: 'burgers', label: 'Burgers', icon: '🍔' },
  { key: 'pizza', label: 'Pizza', icon: '🍕' },
  { key: 'wraps', label: 'Wraps', icon: '🌯' },
  { key: 'sides', label: 'Sides', icon: '🍟' },
  { key: 'beverages', label: 'Beverages', icon: '🥤' },
  { key: 'desserts', label: 'Desserts', icon: '🍰' },
];

/** Categories shown during peak hours (fast-prep, low-queue impact). */
const PEAK_CATEGORIES = new Set(['burgers', 'beverages', 'sides']);

const categoryMeta = (key: string) =>
  CATEGORY_META.find((c) => c.key === key) ?? {
    key,
    label: key.charAt(0).toUpperCase() + key.slice(1),
    icon: '🍽',
  };

interface MenuPanelProps {
  /** When true, only fast-prep peak categories are shown. */
  peakOnly?: boolean;
}

export function MenuPanel({ peakOnly = false }: MenuPanelProps) {
  const [products, setProducts] = useState<Product[] | null>(null);
  const mountedRef = useRef(false);

  useEffect(() => {
    mountedRef.current = true;
    void (async () => {
      const data = await fetchMenu();
      if (mountedRef.current) setProducts(data);
    })();
    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Group products by category, ordered per CATEGORY_META then any extras.
  const grouped = useMemo(() => {
    if (!products) return [];
    const filtered = peakOnly
      ? products.filter((p) => PEAK_CATEGORIES.has(p.category))
      : products;
    const byCat = new Map<string, Product[]>();
    for (const p of filtered) {
      const list = byCat.get(p.category) ?? [];
      list.push(p);
      byCat.set(p.category, list);
    }
    const ordered: string[] = [
      ...CATEGORY_META.map((c) => c.key).filter((k) => byCat.has(k)),
      ...[...byCat.keys()].filter((k) => !CATEGORY_META.some((c) => c.key === k)).sort(),
    ];
    return ordered.map((key) => ({
      ...categoryMeta(key),
      items: (byCat.get(key) ?? []).sort((a, b) => a.name.localeCompare(b.name)),
    }));
  }, [products, peakOnly]);

  if (products === null) {
    return <p className="px-1 py-3 text-sm text-kiosk-textlo">Loading menu…</p>;
  }

  if (products.length === 0) {
    return <p className="px-1 py-3 text-sm text-kiosk-textlo">Menu is currently unavailable.</p>;
  }

  return (
    <div className="space-y-3">
      {grouped.map((cat) => (
        <div
          key={cat.key}
          className="overflow-hidden rounded-lg border border-gray-200 bg-white shadow-sm"
        >
          <div className="flex items-center justify-between border-b border-gray-100 bg-gray-50 px-3 py-2">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-500">
              {cat.icon} {cat.label}
            </span>
            <span className="text-[10px] text-gray-400">{cat.items.length}</span>
          </div>
          <div className="divide-y divide-gray-100">
            {cat.items.map((item) => (
              <div
                key={item.product_id}
                className="flex items-center justify-between gap-3 px-3 py-2"
              >
                <span className="min-w-0 truncate text-sm text-intel-dark">{item.name}</span>
                <span className="shrink-0 text-sm font-medium text-intel-dark">
                  {formatPrice(item.price)}
                </span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

export default MenuPanel;
