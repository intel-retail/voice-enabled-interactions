import { endpoints } from '../constants';

/** Per-file outcome inside a batch ingest response. */
export interface FileIngestResult {
  source?: string;
  chunks_added?: number;
  status?: string;
  detail?: string;
}

/**
 * Response of `POST /api/v1/context/file`.
 *
 * Current builds return `BatchIngestResponse` (`total_chunks_added` + `results`).
 * `chunks_added`/`source` are kept optional for older single-file builds.
 */
export interface IngestResult {
  total_chunks_added?: number;
  files_processed?: number;
  files_succeeded?: number;
  files_failed?: number;
  results?: FileIngestResult[];
  chunks_added?: number | string;
  source?: string;
  [key: string]: unknown;
}

/**
 * Build a user-facing status line from an ingest response, tolerating both the
 * batch and the legacy single-file payload shapes.
 */
export function summariseIngest(
  result: IngestResult,
  fallbackName: string,
): { ok: boolean; message: string } {
  const perFile = result.results ?? [];
  const failed = perFile.filter((r) => r.status && r.status !== 'ok');

  const chunks = Number(
    result.total_chunks_added ??
      result.chunks_added ??
      perFile.reduce((sum, r) => sum + (r.chunks_added ?? 0), 0),
  );
  const source = perFile[0]?.source ?? result.source ?? fallbackName;

  if (failed.length > 0 || (result.files_failed ?? 0) > 0) {
    const detail = failed.map((r) => r.detail || 'unknown error').join('; ');
    return {
      ok: false,
      message: `⚠️ Ingestion failed for ${source}: ${detail || 'unknown error'}. Previous knowledge base remains active.`,
    };
  }

  if (!Number.isFinite(chunks) || chunks <= 0) {
    return {
      ok: false,
      message: `⚠️ Ingestion produced 0 chunks from ${source}. The knowledge base may be empty.`,
    };
  }

  return { ok: true, message: `✅ Knowledge base updated — ${chunks} chunks from ${source}` };
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
