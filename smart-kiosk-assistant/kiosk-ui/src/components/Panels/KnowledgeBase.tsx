import { useState } from 'react';
import type { ChangeEvent } from 'react';

import { fetchSampleFile, ingestDocument } from '../../api/ragApi';
import { sampleKnowledgeBases } from '../../constants';

interface KnowledgeBaseProps {
  onIngestStateChange?: (busy: boolean) => void;
}

type StatusKind = 'idle' | 'loading' | 'success' | 'error' | 'warn';

interface StatusState {
  kind: StatusKind;
  message: string;
}

const statusClasses: Record<StatusKind, string> = {
  idle: '',
  loading: 'text-intel-blue',
  success: 'text-green-600',
  error: 'text-red-600',
  warn: 'text-amber-600',
};

const defaultSample = sampleKnowledgeBases[0]?.file ?? '';

export function KnowledgeBase({ onIngestStateChange }: KnowledgeBaseProps) {
  const [selectedSample, setSelectedSample] = useState(defaultSample);
  const [status, setStatus] = useState<StatusState>({ kind: 'idle', message: '' });
  const [busy, setBusy] = useState(false);

  const runIngest = async (filename: string, content: Blob) => {
    setBusy(true);
    onIngestStateChange?.(true);
    setStatus({ kind: 'loading', message: '⏳ Ingesting knowledge base…' });

    try {
      const result = await ingestDocument(filename, content);
      setStatus({
        kind: 'success',
        message: `✅ Knowledge base updated — ${result.chunks_added ?? 0} chunks from ${result.source ?? filename}`,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      setStatus({
        kind: 'error',
        message: `⚠️ Ingestion failed: ${message}. Previous knowledge base remains active.`,
      });
    } finally {
      setBusy(false);
      onIngestStateChange?.(false);
    }
  };

  const handleSampleIngest = async () => {
    if (!selectedSample) {
      setStatus({ kind: 'warn', message: 'Select a sample knowledge base first.' });
      return;
    }

    setBusy(true);
    onIngestStateChange?.(true);
    setStatus({ kind: 'loading', message: '⏳ Ingesting knowledge base…' });

    try {
      const blob = await fetchSampleFile(selectedSample);
      const result = await ingestDocument(selectedSample, blob);
      setStatus({
        kind: 'success',
        message: `✅ Knowledge base updated — ${result.chunks_added ?? 0} chunks from ${result.source ?? selectedSample}`,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      setStatus({
        kind: 'error',
        message: `⚠️ Ingestion failed: ${message}. Previous knowledge base remains active.`,
      });
    } finally {
      setBusy(false);
      onIngestStateChange?.(false);
    }
  };

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    await runIngest(file.name, file);
    event.target.value = '';
  };

  return (
    <section className="rounded-lg border border-kiosk-border bg-white p-4">
      <p className="mb-3 text-sm text-kiosk-textmd">
        Replace the assistant&apos;s knowledge base with a sample or your own .txt / .md document.
      </p>

      <select
        className="mb-2 w-full rounded-md border border-kiosk-border bg-white px-3 py-2 text-sm"
        disabled={busy}
        value={selectedSample}
        onChange={(event) => setSelectedSample(event.target.value)}
      >
        {sampleKnowledgeBases.map((sample) => (
          <option key={sample.file} value={sample.file}>
            {sample.label}
          </option>
        ))}
      </select>

      <a href={`/samples/${selectedSample}`} download className="text-xs text-intel-blue hover:underline">
        Download selected sample
      </a>

      <div className="mt-3 flex gap-2">
        <button
          type="button"
          className="rounded-md border border-kiosk-border px-3 py-1.5 text-sm text-intel-dark hover:bg-kiosk-pane disabled:opacity-50"
          disabled={busy}
          onClick={() => void handleSampleIngest()}
        >
          Use Sample &amp; Ingest
        </button>

        <label
          aria-disabled={busy}
          className={`rounded-md bg-intel-blue px-3 py-1.5 text-sm text-white hover:bg-intel-blue-dark disabled:opacity-50 ${
            busy ? 'pointer-events-none opacity-50' : ''
          }`}
        >
          📄 Upload .txt / .md &amp; Ingest
          <input
            type="file"
            accept=".txt,.md"
            className="hidden"
            disabled={busy}
            onChange={(event) => void handleFileChange(event)}
          />
        </label>
      </div>

      {status.kind !== 'idle' ? (
        <p className={`mt-3 text-sm ${statusClasses[status.kind]}`}>{status.message}</p>
      ) : null}
    </section>
  );
}

export default KnowledgeBase;
