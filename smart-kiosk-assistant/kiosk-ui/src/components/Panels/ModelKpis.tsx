import type { KpiBundle } from '../../types';

type ModelKpisProps = {
  kpis: KpiBundle;
  onRefresh: () => void;
};

type CardProps = {
  title: string;
  rows: [string, string][];
};

const s = (v: unknown) => (v === null || v === undefined || v === '' ? '—' : String(v));
const ms = (v: unknown) => (typeof v === 'number' ? `${v.toLocaleString()} ms` : '—');
const tail = (v: unknown) => s(v).split('/').pop() ?? '—';

function Card({ title, rows }: CardProps) {
  return (
    <div className="border border-kiosk-border rounded-lg p-3 bg-white">
      <h3 className="text-sm font-semibold text-intel-dark mb-2">{title}</h3>
      <div className="space-y-1">
        {rows.map(([label, value]) => (
          <div key={label} className="flex justify-between gap-3">
            <span className="text-xs text-kiosk-textmd">{label}</span>
            <span className="text-xs font-medium text-intel-dark text-right">{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ModelKpis({ kpis, onRefresh }: ModelKpisProps) {
  const asr = kpis.asr;
  const rag = kpis.rag;
  const tts = kpis.tts;

  const ap = (asr.perf ?? {}) as Record<string, unknown>;
  const rp = (rag.perf ?? {}) as Record<string, unknown>;
  const retr = (rp.retrieval ?? {}) as Record<string, unknown>;
  const llm = (rp.llm ?? {}) as Record<string, unknown>;
  const tp = (tts.perf ?? {}) as Record<string, unknown>;

  return (
    <div className="space-y-3">
      <Card
        title="🎤 ASR — Speech Recognition"
        rows={[
          ['Model', tail(asr.model)],
          ['Backend', s(asr.provider)],
          ['Precision', s(asr.weight_format)],
          ['Device', s(asr.device).toUpperCase()],
          ['Last latency', ms(ap.last_ms)],
        ]}
      />
      <Card
        title="🔍 RAG — Retrieval + Generation"
        rows={[
          ['LLM', tail(rag.llm_model)],
          ['LLM Device', s(rag.llm_device)],
          ['Precision', s(rag.llm_weight_format)],
          ['Embeddings', tail(rag.embedding_model)],
          ['Emb Device', s(rag.embedding_device)],
          ['Reranker', tail(rag.reranker_model)],
          ['Docs indexed', s(rag.document_count)],
          ['Top-K', s(rag.top_k)],
          ['Retrieval lat.', ms(retr.last_ms)],
          ['LLM lat.', ms(llm.last_ms)],
        ]}
      />
      <Card
        title="🔊 TTS — Speech Synthesis"
        rows={[
          ['Model', tail(tts.model)],
          ['Backend', s(tts.runtime)],
          ['Precision', s(tts.dtype)],
          ['Device', s(tts.device).toUpperCase()],
          ['Language', s(tts.default_language)],
          ['Last latency', ms(tp.last_ms)],
        ]}
      />
      <button
        type="button"
        className="text-sm px-3 py-1.5 rounded-md border border-kiosk-border text-intel-dark hover:bg-kiosk-pane"
        onClick={onRefresh}
      >
        🔄 Refresh
      </button>
    </div>
  );
}
