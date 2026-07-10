import { endpoints } from '../constants';

export interface IngestResult {
  chunks_added?: number | string;
  source?: string;
  [key: string]: unknown;
}

/**
 * Clear the existing knowledge base, then ingest a new document.
 * Mirrors the Gradio ingest flow: DELETE context → POST context/file.
 */
export async function ingestDocument(filename: string, content: Blob): Promise<IngestResult> {
  // 1. Wipe the existing knowledge base (best-effort).
  try {
    await fetch(endpoints.ragContext, { method: 'DELETE' });
  } catch {
    /* non-fatal — proceed to ingest */
  }

  // 2. Ingest the new document.
  const form = new FormData();
  form.append('file', new File([content], filename, { type: 'text/plain' }));

  const res = await fetch(endpoints.ragContextFile, { method: 'POST', body: form });
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      detail = err.detail || err.error || detail;
    } catch {
      /* keep default */
    }
    throw new Error(String(detail));
  }
  return res.json();
}

/** Fetch a built-in sample knowledge-base markdown file from the SPA assets. */
export async function fetchSampleFile(file: string): Promise<Blob> {
  const res = await fetch(`/samples/${file}`);
  if (!res.ok) throw new Error(`Failed to load sample ${file}: ${res.status}`);
  return res.blob();
}
