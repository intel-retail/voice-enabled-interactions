import { endpoints, tuning } from '../constants';
import type { HistoryTurn, SessionSnapshot, StartStreamResponse } from '../types';

/**
 * Backend service URLs for kiosk-core to call.
 * These are internal Docker network URLs (not the nginx-proxied paths).
 * kiosk-core runs inside the Docker network and calls these services directly.
 */
const BACKEND_SERVICE_URLS = {
  analyzer_url: 'http://audio-analyzer:8010/v1/audio/transcriptions',
  rag_url: 'http://rag-service:8020/api/v1/query',
  tts_url: 'http://text-to-speech:8011/v1/audio/speech',
};

/** Open a new browser streaming session on kiosk-core. */
export async function startStreamSession(
  sampleRate: number,
  history: HistoryTurn[],
): Promise<StartStreamResponse> {
  const res = await fetch(endpoints.startStream, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sample_rate: sampleRate,
      chunk_seconds: tuning.chunkSeconds,
      silence_timeout_seconds: 2.0,
      max_session_seconds: 60.0,
      silence_threshold: 900,
      language: 'en',
      temperature: 0.0,
      tts_model: 'speecht5',
      tts_language: 'English',
      history,
      // Backend service URLs (kiosk-core will call these internally)
      ...BACKEND_SERVICE_URLS,
    }),
  });
  if (!res.ok) throw new Error(`Failed to start session: ${res.status} ${res.statusText}`);
  return res.json();
}

/** Push a 16-bit mono PCM WAV chunk into an active session. */
export async function pushAudioChunk(sessionId: string, wav: Blob): Promise<void> {
  const res = await fetch(endpoints.pushAudio(sessionId), {
    method: 'POST',
    headers: { 'Content-Type': 'audio/wav' },
    body: wav,
  });
  if (!res.ok) throw new Error(`Failed to push audio: ${res.status}`);
}

/** Signal end-of-stream so the session finalises (RAG + TTS). */
export async function endAudioStream(sessionId: string): Promise<void> {
  const res = await fetch(endpoints.endAudio(sessionId), { method: 'POST' });
  if (!res.ok) throw new Error(`Failed to end stream: ${res.status}`);
}

/** Poll the current session snapshot (transcript, response, tts segments, status). */
export async function pollSession(sessionId: string): Promise<SessionSnapshot> {
  const res = await fetch(endpoints.pollSession(sessionId));
  if (!res.ok) throw new Error(`Failed to poll session: ${res.status}`);
  return res.json();
}

/** Build the URL the browser uses to fetch a generated TTS wav. */
export function ttsAudioUrl(sessionId: string, absoluteServerPath: string): string {
  const filename = absoluteServerPath.split('/').pop() ?? '';
  return endpoints.sessionAudioFile(sessionId, filename);
}
