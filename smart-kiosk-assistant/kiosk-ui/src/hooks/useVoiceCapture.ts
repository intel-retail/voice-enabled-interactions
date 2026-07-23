import { useCallback, useRef, useState } from 'react';
import { tuning } from '../constants';
import { concatFloat32, encodeWav, resampleLinear } from '../api/audioUtils';

const TARGET_RATE = tuning.sampleRate; // 16000

// Voice-activity heuristics: stop early once a full sentence has been
// spoken (sustained silence after detected speech), otherwise fall back to
// the hard `maxSeconds` cap so the capture never hangs indefinitely.
const SPEECH_RMS_THRESHOLD = 0.015;
const MIN_SPEECH_MS = 1200;
const TRAILING_SILENCE_MS = 1500;

interface UseVoiceCaptureResult {
  recording: boolean;
  error: string | null;
  /**
   * Record a mono WAV clip and resolve to base64 (no `data:` prefix).
   * Recording stops automatically once a sentence has been spoken
   * (speech followed by ~1.5s of silence), or after `maxSeconds` elapses,
   * whichever comes first.
   */
  recordClip: (maxSeconds?: number) => Promise<string | null>;
}

/**
 * Voice-activity-aware capture for the identity login/register challenge
 * phrase. Reuses the same AudioWorklet PCM pipeline as `useVoiceSession`
 * (`/pcm-capture-processor.js`) and the shared `audioUtils` WAV encoder, but
 * captures a single clip instead of streaming chunks. Recording ends as soon
 * as the spoken sentence trails off into silence, giving the ECAPA voice
 * embedding a fuller, more consistent sample than a fixed short clip.
 */
export function useVoiceCapture(): UseVoiceCaptureResult {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const stopRef = useRef<(() => void) | null>(null);

  const recordClip = useCallback(async (maxSeconds: number = 5): Promise<string | null> => {
    setError(null);
    setRecording(true);

    let stream: MediaStream | null = null;
    let ctx: AudioContext | null = null;
    let worklet: AudioWorkletNode | null = null;
    let source: MediaStreamAudioSourceNode | null = null;
    const frames: Float32Array[] = [];

    const teardown = () => {
      try {
        worklet?.disconnect();
        source?.disconnect();
      } catch {
        /* ignore */
      }
      stream?.getTracks().forEach((t) => t.stop());
      if (ctx && ctx.state !== 'closed') ctx.close().catch(() => undefined);
    };
    stopRef.current = teardown;

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Microphone access requires HTTPS or localhost.');
      }
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      ctx = new AudioContext();
      await ctx.audioWorklet.addModule('/pcm-capture-processor.js');

      source = ctx.createMediaStreamSource(stream);
      worklet = new AudioWorkletNode(ctx, 'pcm-capture-processor');

      let speechStarted = false;
      let speechStartMs: number | null = null;
      let silenceStartMs: number | null = null;
      let resolveEarlyStop: (() => void) | null = null;
      const earlyStop = new Promise<void>((resolve) => {
        resolveEarlyStop = resolve;
      });

      // Track the sample-offset range that actually contains speech so the
      // trailing/leading silence (including the ~1.5s of silence used just
      // to *detect* the sentence has ended) can be trimmed before encoding —
      // otherwise it dilutes the voice embedding with non-speech samples.
      let samplesSeen = 0;
      let speechStartSample: number | null = null;
      let lastSpeechEndSample = 0;

      worklet.port.onmessage = (ev: MessageEvent<Float32Array>) => {
        const chunk = ev.data;
        frames.push(chunk);

        let sumSq = 0;
        for (let i = 0; i < chunk.length; i++) sumSq += chunk[i] * chunk[i];
        const rms = Math.sqrt(sumSq / chunk.length);
        const now = performance.now();

        if (rms >= SPEECH_RMS_THRESHOLD) {
          if (!speechStarted) {
            speechStarted = true;
            speechStartMs = now;
            speechStartSample = samplesSeen;
          }
          silenceStartMs = null;
          lastSpeechEndSample = samplesSeen + chunk.length;
        } else if (speechStarted) {
          if (silenceStartMs === null) {
            silenceStartMs = now;
          } else if (
            now - silenceStartMs >= TRAILING_SILENCE_MS &&
            now - (speechStartMs ?? now) >= MIN_SPEECH_MS
          ) {
            resolveEarlyStop?.();
          }
        }
        samplesSeen += chunk.length;
      };
      source.connect(worklet);
      worklet.connect(ctx.destination);

      const ctxRate = ctx.sampleRate;
      const hardCap = new Promise<void>((resolve) => setTimeout(resolve, maxSeconds * 1000));
      await Promise.race([earlyStop, hardCap]);

      teardown();
      let merged = concatFloat32(frames);

      // Trim leading/trailing silence, keeping a small padding margin around
      // the detected speech so words aren't clipped.
      if (speechStartSample !== null) {
        const padSamples = Math.round(ctxRate * 0.2); // 200ms pre/post-roll
        const start = Math.max(0, speechStartSample - padSamples);
        const end = Math.min(merged.length, lastSpeechEndSample + padSamples);
        if (end > start) {
          merged = merged.slice(start, end);
        }
      }

      const resampled = resampleLinear(merged, ctxRate, TARGET_RATE);
      const wav = encodeWav(resampled, TARGET_RATE);
      const buf = await wav.arrayBuffer();
      let binary = '';
      const bytes = new Uint8Array(buf);
      for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
      return btoa(binary);
    } catch (err) {
      teardown();
      const msg = err instanceof Error ? err.message : String(err);
      setError(`Unable to capture audio: ${msg}`);
      return null;
    } finally {
      setRecording(false);
      stopRef.current = null;
    }
  }, []);

  return { recording, error, recordClip };
}
