import { endpoints } from '../constants';
import type { ChartPoint, MetricSeriesItem, MetricsResponse } from '../types';

/**
 * Fetch time-series metrics from the metrics-collector service.
 *
 * Throws on transport failure, timeout or a non-OK response. Callers must
 * surface that rather than substituting an empty object: an unreachable
 * collector and a collector with no samples yet both render as blank charts,
 * which previously made outages effectively undiagnosable from the UI.
 */
export async function fetchMetrics(): Promise<MetricsResponse> {
  const res = await fetch(endpoints.metrics, { signal: AbortSignal.timeout(4000) });
  if (!res.ok) {
    throw new Error(`metrics-collector returned HTTP ${res.status}`);
  }
  return (await res.json()) as MetricsResponse;
}

/** Convert a [[ts, value, ...], ...] series into recharts points. */
export function toChartPoints(
  series: MetricSeriesItem[] | undefined,
  valueIndex = 1,
): ChartPoint[] {
  if (!series) return [];
  return series.map((item) => {
    let time = String(item[0]);
    try {
      const d = new Date(item[0]);
      if (!Number.isNaN(d.getTime())) {
        time = d.toLocaleTimeString('en-GB', { hour12: false });
      }
    } catch {
      /* keep raw */
    }
    const value = item.length > valueIndex ? Number(item[valueIndex]) : 0;
    return { time, value: Number.isFinite(value) ? value : 0 };
  });
}
