import { useCallback, useEffect, useRef, useState } from 'react';

import { fetchKpis } from '../api/ttsApi';
import type { KpiBundle } from '../types';

const initialKpis = (): KpiBundle => ({ asr: {}, rag: {}, tts: {} });

export function useKpis(): { kpis: KpiBundle; loading: boolean; refresh: () => void } {
  const mountedRef = useRef(false);
  const [kpis, setKpis] = useState<KpiBundle>(() => initialKpis());
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(() => {
    if (!mountedRef.current) return;

    setLoading(true);
    void fetchKpis()
      .then((nextKpis) => {
        if (mountedRef.current) {
          setKpis(nextKpis);
        }
      })
      .catch(() => undefined)
      .finally(() => {
        if (mountedRef.current) {
          setLoading(false);
        }
      });
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    refresh();

    return () => {
      mountedRef.current = false;
    };
  }, [refresh]);

  return { kpis, loading, refresh };
}
