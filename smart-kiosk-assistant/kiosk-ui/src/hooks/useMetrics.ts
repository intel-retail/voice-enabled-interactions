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
  /** Epoch ms of the last successful poll, or null if none has succeeded.
   *  Lets the UI show how stale the retained sample is during an outage. */
  lastUpdated: number | null;
  refresh: () => void;
}

export function useMetrics(): UseMetricsResult {
  const [metrics, setMetrics] = useState<MetricsResponse>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const isMountedRef = useRef<boolean>(false);
  const isFetchingRef = useRef<boolean>(false);
  // Tracks whether the previous poll failed, so an ongoing outage is logged
  // once on the healthy -> failing edge instead of on every poll interval.
  const wasFailingRef = useRef<boolean>(false);

  const loadMetrics = useCallback(async (): Promise<void> => {
    if (isFetchingRef.current) return;

    isFetchingRef.current = true;
    if (isMountedRef.current) setLoading(true);

    try {
      const nextMetrics = await fetchMetrics();
      if (wasFailingRef.current) {
        console.info('[useMetrics] metrics poll recovered');
      }
      wasFailingRef.current = false;
      if (isMountedRef.current) {
        setMetrics(nextMetrics);
        setError(null);
        setLastUpdated(Date.now());
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      // Keep the last good sample on screen rather than blanking the charts,
      // and make the failure visible instead of silently showing no data.
      // Log only on the transition into the failing state: polling every
      // perfRefreshMs would otherwise flood the console for the whole outage.
      if (!wasFailingRef.current) {
        console.error('[useMetrics] metrics poll failed:', message);
        wasFailingRef.current = true;
      }
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

  return { metrics, loading, error, lastUpdated, refresh };
}
