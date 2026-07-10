import { useCallback, useEffect, useRef, useState } from 'react';

import { fetchMetrics } from '../api/metricsApi';
import { tuning } from '../constants';
import type { MetricsResponse } from '../types';

interface UseMetricsResult {
  metrics: MetricsResponse;
  loading: boolean;
  refresh: () => void;
}

export function useMetrics(): UseMetricsResult {
  const [metrics, setMetrics] = useState<MetricsResponse>({});
  const [loading, setLoading] = useState<boolean>(true);
  const isMountedRef = useRef<boolean>(false);
  const isFetchingRef = useRef<boolean>(false);

  const loadMetrics = useCallback(async (): Promise<void> => {
    if (isFetchingRef.current) return;

    isFetchingRef.current = true;
    if (isMountedRef.current) setLoading(true);

    try {
      const nextMetrics = await fetchMetrics();
      if (isMountedRef.current) {
        setMetrics(nextMetrics);
      }
    } finally {
      isFetchingRef.current = false;
      if (isMountedRef.current) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    isMountedRef.current = true;
    void loadMetrics();

    const intervalId = window.setInterval(() => {
      void loadMetrics();
    }, tuning.perfRefreshMs);

    return () => {
      isMountedRef.current = false;
      window.clearInterval(intervalId);
    };
  }, [loadMetrics]);

  const refresh = useCallback((): void => {
    void loadMetrics();
  }, [loadMetrics]);

  return { metrics, loading, refresh };
}
