import { endpoints } from '../constants';
import type { KpiBundle, KpiData } from '../types';

async function getJson(url: string): Promise<Record<string, unknown>> {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return {};
    return await res.json();
  } catch {
    return {};
  }
}

/**
 * Fetch merged model-info + latency for ASR, RAG and TTS services.
 * Mirrors gradio_app _fetch_kpis().
 */
export async function fetchKpis(): Promise<KpiBundle> {
  const [asrInfo, asrPerf, ttsInfo, ttsPerf, ragInfo, ragPerf] = await Promise.all([
    getJson(endpoints.asrModelInfo),
    getJson(endpoints.asrPerformance),
    getJson(endpoints.ttsModelInfo),
    getJson(endpoints.ttsPerformance),
    getJson(endpoints.ragModelInfo),
    getJson(endpoints.ragPerformance),
  ]);

  const merge = (info: Record<string, unknown>, perf: Record<string, unknown>): KpiData => ({
    ...info,
    perf: (perf.latency as Record<string, unknown>) ?? {},
  });

  return {
    asr: merge(asrInfo, asrPerf),
    rag: merge(ragInfo, ragPerf),
    tts: merge(ttsInfo, ttsPerf),
  };
}
