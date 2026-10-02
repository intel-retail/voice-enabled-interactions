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
  conversationId?: string,
  singleChunk = false,
): Promise<StartStreamResponse> {
  // Push-to-talk uploads audio to the backend continuously in small slices
  // (see flushChunk in useVoiceSession.ts) so the server's VAD/frame pipeline
  // never backlogs, but it must still be transcribed as ONE uncut Whisper
  // call over the whole utterance, not re-split at upload boundaries. So the
  // backend's own chunk cap and silence endpoint are pushed beyond any
  // realistic utterance, leaving the explicit end-of-stream signal (Stop) as
  // the only thing that triggers ASR and ends the turn. Hands-free
  // conversation mode keeps the tuned values: it has no Stop button and
  // relies on silence endpointing to end a turn.
  const chunkSeconds = singleChunk
    ? tuning.singleChunkMaxSeconds
    : tuning.asrChunkSeconds;
  const silenceTimeout = singleChunk
    ? tuning.singleChunkSilenceSeconds
    : tuning.silenceTimeoutSeconds;
  const res = await fetch(endpoints.startStream, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sample_rate: sampleRate,
      chunk_seconds: chunkSeconds,
      silence_timeout_seconds: silenceTimeout,
      // The adaptive pre-warm flush fires on any pause >= 0.70s and would
      // split the recording exactly like the chunk cap does, so raising the
      // two values above is not sufficient on its own — it must be switched
      // off for single-chunk mode. It is purely a latency optimisation
      // (it starts ASR early); with one upload at Stop there is no speech
      // left to overlap with, so nothing is lost by disabling it.
      ...(singleChunk ? { adaptive_flush_pause_seconds: 0 } : {}),
      // Equal to chunk_seconds in single-chunk mode so a recording can never
      // exceed one chunk (see tuning.singleChunkMaxSeconds).
      max_session_seconds: singleChunk ? tuning.singleChunkMaxSeconds : 60.0,
      silence_threshold: 900,
      language: 'en',
      temperature: 0.0,
      // tts_model intentionally omitted: kiosk-core's SessionStartRequest
      // defaults it to config.DEFAULT_TTS_MODEL (server-side, currently
      // "kokoro" via KIOSK_CORE_TTS_MODEL in docker-compose.yml). This used
      // to hardcode 'speecht5', which silently overrode that default and
      // kept every browser session on the old backend even after the TTS
      // service switched to Kokoro.
      tts_language: 'English',
      history,
      // Persistent conversation ID — reused across all voice turns so the
      // agent retains cart/order state between microphone presses.
      ...(conversationId ? { conversation_id: conversationId } : {}),
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
