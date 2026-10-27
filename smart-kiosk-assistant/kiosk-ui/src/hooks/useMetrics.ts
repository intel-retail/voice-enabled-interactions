import { useCallback, useEffect, useRef, useState } from 'react';

import { fetchMetrics } from '../api/metricsApi';
import { tuning } from '../constants';
import type { MetricsResponse } from '../types';

interface UseMetricsResult {
  metrics: MetricsResponse;
  loading: boolean;
  /** Non-null when the most recent poll failed; distinguishes an unreachable
   *  collector from a reachable one that has no samples yet. */
  error: string | null;
  refresh: () => void;
}

export function useMetrics(): UseMetricsResult {
  const [metrics, setMetrics] = useState<MetricsResponse>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
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
        setError(null);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      // Keep the last good sample on screen rather than blanking the charts,
      // and make the failure visible instead of silently showing no data.
      console.error('[useMetrics] metrics poll failed:', message);
      if (isMountedRef.current) {
        setError(message);
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

  return { metrics, loading, error, refresh };
}
