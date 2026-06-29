import { useCallback, useEffect, useRef, useState } from 'react';
import { tuning } from '../constants';
import {
  endAudioStream,
  pollSession,
  pushAudioChunk,
  startStreamSession,
  ttsAudioUrl,
} from '../api/kioskApi';
import { concatFloat32, encodeWav, resampleLinear } from '../api/audioUtils';
import { useAudioQueue } from './useAudioQueue';
import type { ChatMessage, HistoryTurn, VoicePhase } from '../types';

interface UseVoiceSessionOptions {
  deviceId: string;
  enabled: boolean; // false while a knowledge-base ingest is in progress
  onTurnComplete?: () => void;
}

const TARGET_RATE = tuning.sampleRate; // 16000

export function useVoiceSession({ deviceId, enabled, onTurnComplete }: UseVoiceSessionOptions) {
  const [phase, setPhase] = useState<VoicePhase>('idle');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [partialUser, setPartialUser] = useState('');
  const [partialAssistant, setPartialAssistant] = useState('');
  const [statusText, setStatusText] = useState('Tap the mic and ask a question');
  const [error, setError] = useState<string | null>(null);

  const audioQueue = useAudioQueue({
    onAllDone: () => {
      // After the assistant finishes speaking, allow a KPI/order refresh.
      onTurnComplete?.();
    },
  });

  // ── Mutable streaming refs ────────────────────────────────────────────────
  const ctxRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const framesRef = useRef<Float32Array[]>([]);
  const ctxRateRef = useRef<number>(48000);
  const sessionIdRef = useRef<string | null>(null);
  const recordingRef = useRef(false);
  const eosRef = useRef(false);
  const pollTimerRef = useRef<number | null>(null);
  const messagesRef = useRef<ChatMessage[]>([]);
  messagesRef.current = messages;

  const buildHistory = useCallback((): HistoryTurn[] => {
    const recent = messagesRef.current.slice(-tuning.maxHistoryTurns);
    return recent
      .filter((m) => m.text.trim())
      .map((m) => ({ role: m.role, content: m.text }));
  }, []);

  const flushChunk = useCallback(async (force = false) => {
    const frames = framesRef.current;
    if (frames.length === 0) return;
    const total = frames.reduce((acc, f) => acc + f.length, 0);
    const haveSeconds = total / ctxRateRef.current;
    if (!force && haveSeconds < tuning.chunkSeconds) return;

    framesRef.current = [];
    const merged = concatFloat32(frames);
    const resampled = resampleLinear(merged, ctxRateRef.current, TARGET_RATE);
    const wav = encodeWav(resampled, TARGET_RATE);

    const sid = sessionIdRef.current;
    if (!sid) return;
    try {
      await pushAudioChunk(sid, wav);
    } catch {
      /* transient push error — drop this chunk */
    }
  }, []);

  const teardownCapture = useCallback(() => {
    recordingRef.current = false;
    try {
      workletRef.current?.disconnect();
      sourceRef.current?.disconnect();
    } catch {
      /* ignore */
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    if (ctxRef.current && ctxRef.current.state !== 'closed') {
      ctxRef.current.close().catch(() => undefined);
    }
    workletRef.current = null;
    sourceRef.current = null;
    streamRef.current = null;
    ctxRef.current = null;
  }, []);

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current !== null) {
      window.clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  // ── Single poll loop: drives partial transcript, response, TTS, completion ──
  const pollLoop = useCallback(async () => {
    const sid = sessionIdRef.current;
    if (!sid) return;

    let snapshot;
    try {
      snapshot = await pollSession(sid);
    } catch {
      pollTimerRef.current = window.setTimeout(pollLoop, tuning.pollIntervalMs);
      return;
    }

    const transcript = (snapshot.transcript ?? '').trim();
    const response = (snapshot.response ?? '').trim();
    const running = snapshot.status === 'running' || snapshot.status === 'stopping';

    if (transcript) setPartialUser(transcript);
    if (response) setPartialAssistant(response);

    // Enqueue any new TTS audio segments.
    const segs = snapshot.tts_audio_segments ?? [];
    if (segs.length > 0) {
      const urls = segs.map((s) => ttsAudioUrl(sid, String(s.audio_file)));
      audioQueue.enqueue(urls);
    }

    // Status text mirrors the Gradio state machine.
    if (eosRef.current) {
      if (segs.length) setStatusText(`🔊 Speaking… (${segs.length})`);
      else if (response) setStatusText('💬 Generating response…');
      else if (transcript) setStatusText('📝 Querying knowledge base…');
      else setStatusText('⏳ Processing speech…');
    }

    // Completion: EOS signalled and the backend has finished.
    if (eosRef.current && !running) {
      stopPolling();
      const finalTranscript = transcript;
      const finalResponse = response;
      setMessages((prev) => {
        const next = [...prev];
        if (finalTranscript) next.push({ role: 'user', text: finalTranscript });
        if (finalResponse) next.push({ role: 'assistant', text: finalResponse });
        return next;
      });
      setPartialUser('');
      setPartialAssistant('');
      sessionIdRef.current = null;
      eosRef.current = false;
      setPhase('idle');
      setStatusText('✓ Done — tap 🎤 for another question');
      // onTurnComplete also fires when TTS finishes (onAllDone); fire here too
      // in case there was no audio to play.
      if (segs.length === 0) onTurnComplete?.();
      return;
    }

    pollTimerRef.current = window.setTimeout(pollLoop, tuning.pollIntervalMs);
  }, [audioQueue, onTurnComplete, stopPolling]);

  // ── Public: start recording ───────────────────────────────────────────────
  const start = useCallback(async () => {
    if (!enabled) {
      setStatusText('⏳ Ingestion in progress — please wait…');
      return;
    }
    if (recordingRef.current || phase !== 'idle') return;
    setError(null);
    audioQueue.reset();
    framesRef.current = [];
    eosRef.current = false;
    setPartialUser('🎤 Listening…');
    setPartialAssistant('');
    setPhase('listening');
    setStatusText('🎙 Listening — speak now');

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Microphone access requires HTTPS or localhost.');
      }
      const constraints: MediaStreamConstraints = {
        audio: deviceId ? { deviceId: { exact: deviceId } } : true,
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      const ctx = new AudioContext();
      ctxRef.current = ctx;
      ctxRateRef.current = ctx.sampleRate;
      await ctx.audioWorklet.addModule('/pcm-capture-processor.js');

      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source;
      const worklet = new AudioWorkletNode(ctx, 'pcm-capture-processor');
      workletRef.current = worklet;

      worklet.port.onmessage = (ev: MessageEvent<Float32Array>) => {
        if (!recordingRef.current) return;
        framesRef.current.push(ev.data);
        void flushChunk(false);
      };

      source.connect(worklet);
      // Connect to destination so the worklet's process() runs; output is silent.
      worklet.connect(ctx.destination);
      recordingRef.current = true;

      // Open the streaming session up-front to avoid first-chunk races.
      const { session_id } = await startStreamSession(TARGET_RATE, buildHistory());
      sessionIdRef.current = session_id;

      // Begin the poll loop (partial transcript while listening).
      stopPolling();
      pollTimerRef.current = window.setTimeout(pollLoop, tuning.pollIntervalMs);
    } catch (err) {
      teardownCapture();
      setPhase('idle');
      setPartialUser('');
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setStatusText(`❌ ${msg}`);
    }
  }, [
    enabled,
    phase,
    deviceId,
    audioQueue,
    buildHistory,
    flushChunk,
    pollLoop,
    stopPolling,
    teardownCapture,
  ]);

  // ── Public: stop recording → finalise ─────────────────────────────────────
  const stop = useCallback(async () => {
    if (!recordingRef.current) return;
    recordingRef.current = false;
    setPhase('processing');
    setStatusText('⏳ Processing…');
    setPartialUser((p) => (p === '🎤 Listening…' ? '⏳ Processing…' : p));

    // Flush remaining audio, then signal end-of-stream.
    await flushChunk(true);
    teardownCapture();

    const sid = sessionIdRef.current;
    if (!sid) {
      setPhase('idle');
      setStatusText('No audio — try again');
      setPartialUser('');
      return;
    }
    try {
      await endAudioStream(sid);
      eosRef.current = true;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setError(msg);
      setStatusText(`❌ ${msg}`);
      setPhase('idle');
      return;
    }
    // The poll loop (already running) will detect completion.
  }, [flushChunk, teardownCapture]);

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      stopPolling();
      teardownCapture();
      audioQueue.stop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    phase,
    messages,
    partialUser,
    partialAssistant,
    statusText,
    error,
    playbackState: audioQueue.state,
    start,
    stop,
  };
}
