import { endpoints } from '../constants';
import type { KpiBundle, KpiData, PipelineTurnTrace } from '../types';

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
 * Also fetches the latest pipeline turn trace from kiosk-core.
 */
export async function fetchKpis(): Promise<KpiBundle> {
  const [asrInfo, asrPerf, ttsInfo, ttsPerf, ragInfo, ragPerf, pipelineData, pipelineRecentData] =
    await Promise.all([
      getJson(endpoints.asrModelInfo),
      getJson(endpoints.asrPerformance),
      getJson(endpoints.ttsModelInfo),
      getJson(endpoints.ttsPerformance),
      getJson(endpoints.ragModelInfo),
      getJson(endpoints.ragPerformance),
      getJson(endpoints.pipelineLatest),
      getJson(endpoints.pipelineRecent),
    ]);

  const merge = (info: Record<string, unknown>, perf: Record<string, unknown>): KpiData => ({
    ...info,
    perf: (perf.latency as Record<string, unknown>) ?? {},
  });

  const pipeline = (pipelineData.trace as PipelineTurnTrace | null | undefined) ?? null;
  const traces = pipelineRecentData.traces;
  const pipelineRecent = Array.isArray(traces) ? (traces as PipelineTurnTrace[]) : [];

  return {
    asr: merge(asrInfo, asrPerf),
    rag: merge(ragInfo, ragPerf),
    tts: merge(ttsInfo, ttsPerf),
    pipeline,
    pipelineRecent,
  };
}
