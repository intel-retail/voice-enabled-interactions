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

// getUserMedia never settles while a permission prompt sits unanswered, and
// some Windows device stacks stall indefinitely too. Without a bound the hook
// parks in "preparing" forever: `recordingRef` is still false so
// endConversation has nothing to finalise, while `phase` is already
// 'listening' so every later start() trips its own `phase !== 'idle'` guard —
// a dead mic button that no amount of tapping can recover. Bound the wait.
const MIC_ACQUIRE_TIMEOUT_MS = 15000;

const MIC_TIMEOUT_MESSAGE =
  'Microphone did not respond. Check the browser permission prompt (the camera/mic icon in the address bar), then try again.';

function withTimeout<T>(promise: Promise<T>, ms: number, message: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error(message)), ms);
    promise.then(
      (value) => {
        window.clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        window.clearTimeout(timer);
        reject(err);
      },
    );
  });
}

/**
 * Acquire the microphone with a bound wait, disposing a late-resolving
 * stream instead of leaking it.
 *
 * `withTimeout` only rejects its own wrapper promise -- it cannot cancel the
 * underlying `getUserMedia()` call. If the permission prompt is answered
 * after the timeout has already fired (and the caller has moved on, e.g. to
 * an error state), `getUserMedia()` still resolves with a live
 * `MediaStream` later; without this, nothing ever stops its tracks, so the
 * mic (and its OS-level "in use" indicator) stays live indefinitely.
 */
function acquireMicWithTimeout(
  constraints: MediaStreamConstraints,
  ms: number,
  message: string,
): Promise<MediaStream> {
  let settled = false;
  const micPromise = navigator.mediaDevices.getUserMedia(constraints);
  micPromise.then(
    (stream) => {
      if (!settled) return; // still within the timeout window -- withTimeout below owns it
      stream.getTracks().forEach((t) => t.stop());
    },
    () => {
      /* rejection is handled by withTimeout below; nothing to dispose */
    },
  );
  return withTimeout(micPromise, ms, message).finally(() => {
    settled = true;
  });
}

/** Map a raw getUserMedia DOMException onto something a customer can act on. */
function describeMicError(err: unknown): string {
  const name = (err as { name?: string } | null)?.name;
  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Microphone permission was blocked. Allow it via the address-bar icon, then try again.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No microphone was found. Connect one and try again.';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'The microphone is in use by another application. Close it and try again.';
    case 'OverconstrainedError':
      return 'The selected microphone is unavailable. Pick a different device in Settings.';
    default:
      return err instanceof Error ? err.message : String(err);
  }
}

export function useVoiceSession({ deviceId, enabled, onTurnComplete }: UseVoiceSessionOptions) {
  const [phase, setPhase] = useState<VoicePhase>('idle');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [partialUser, setPartialUser] = useState('');
  const [partialAssistant, setPartialAssistant] = useState('');
  const [statusText, setStatusText] = useState('Tap the mic and ask a question');
  const [error, setError] = useState<string | null>(null);
  // Hands-free "continuous conversation" mode: once on, a completed turn
  // automatically re-arms the mic instead of returning to idle. Exposed as
  // state (not just a ref) purely so the UI can show "listening…"/"End
  // conversation" instead of the push-to-talk labels.
  const [conversationMode, setConversationMode] = useState(false);
  const conversationModeRef = useRef(false);
  // How many consecutive turns in a row produced no speech at all — guards
  // against an empty room looping forever on ambient noise/silence.
  const noSpeechStreakRef = useRef(0);

  // `start` is defined further down (it needs `audioQueue`, `pollLoop`, etc.)
  // but `audioQueue`'s onAllDone callback needs to call it to auto-resume
  // listening once TTS finishes. A ref breaks that circular dependency.
  const startRef = useRef<(() => Promise<void>) | null>(null);

  const audioQueue = useAudioQueue({
    onAllDone: () => {
      // After the assistant finishes speaking, allow a KPI/order refresh.
      onTurnComplete?.();
      // Hands-free loop: the mic was torn down before TTS started (see the
      // completion branch in pollLoop), so there is no echo risk in resuming
      // now that playback has finished.
      if (conversationModeRef.current) {
        void startRef.current?.();
      }
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
  // Persistent conversation ID — generated once per hook mount and reused
  // across all voice turns so the agent retains order context (cart, order_id).
  const conversationIdRef = useRef<string>(crypto.randomUUID());
  const recordingRef = useRef(false);
  const eosRef = useRef(false);
  // Bumped by every start() and every forceIdle(). An in-flight start()
  // compares the generation it captured against this after each await and
  // bails out if it has been superseded, so a slow getUserMedia that finally
  // resolves after the user gave up cannot resurrect a torn-down session.
  const startGenRef = useRef(0);
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
    // Nothing can be sent until the session exists. Crucially we must NOT
    // drain framesRef before knowing that: audio recorded between the mic
    // going live and startStreamSession resolving is still real speech, and
    // clearing it here used to destroy the opening words of the sentence.
    const sid = sessionIdRef.current;
    if (!sid) return;

    const frames = framesRef.current;
    if (frames.length === 0) return;
    const total = frames.reduce((acc, f) => acc + f.length, 0);
    const haveSeconds = total / ctxRateRef.current;
    if (!force && haveSeconds < tuning.chunkSeconds) return;

    framesRef.current = [];
    const merged = concatFloat32(frames);
    const resampled = resampleLinear(merged, ctxRateRef.current, TARGET_RATE);
    const wav = encodeWav(resampled, TARGET_RATE);

    try {
      await pushAudioChunk(sid, wav);
    } catch {
      // A transient push failure must not silently swallow several seconds of
      // speech. Put the audio back at the head of the buffer so the next flush
      // retries it; ordering is preserved because frames captured meanwhile
      // were appended after this batch was taken.
      framesRef.current = [merged, ...framesRef.current];
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

  // Hard reset back to a tappable idle state, usable from ANY phase.
  // Deliberately unconditional: the previous recovery paths all keyed off
  // `recordingRef.current` or `phase === 'idle'`, so a start() stalled in mic
  // acquisition (recording not yet true, phase already 'listening') matched
  // neither and left the button permanently inert. Bumping the generation
  // also cancels that in-flight start().
  const forceIdle = useCallback(() => {
    startGenRef.current += 1;
    recordingRef.current = false;
    eosRef.current = false;
    sessionIdRef.current = null;
    stopPolling();
    teardownCapture();
    setPartialUser('');
    setPartialAssistant('');
    setPhase('idle');
  }, [stopPolling, teardownCapture]);

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
    if (eosRef.current || (conversationModeRef.current && !running)) {
      if (segs.length) setStatusText(`🔊 Speaking… (${segs.length})`);
      else if (response) setStatusText('💬 Generating response…');
      else if (transcript) setStatusText('📝 Querying knowledge base…');
      else setStatusText('⏳ Processing speech…');
    }

    // Completion: either the frontend explicitly ended the stream (manual
    // push-to-talk), or the backend's own silence-timeout VAD ended the turn
    // on its own (hands-free conversation mode never calls endAudioStream —
    // see startConversation). Either way, once the session is no longer
    // running the turn is done and must be finalised the same way.
    if (!running && (eosRef.current || conversationModeRef.current)) {
      stopPolling();
      // The backend may have ended this turn on its own (silence-timeout VAD)
      // without the frontend ever calling `stop()` — the mic/AudioContext/
      // worklet are still live and `recordingRef` is still true in that case.
      // Tear them down now so the next `start()` (manual or auto-resume) is
      // not silently ignored by its `recordingRef.current` guard, and so the
      // browser doesn't keep sending frames into a session that's already gone.
      recordingRef.current = false;
      teardownCapture();
      const finalTranscript = transcript;
      const finalResponse = response;
      const hadSpeech = Boolean(finalTranscript);
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

      if (conversationModeRef.current) {
        // Guard against looping forever on an empty room: if several turns in
        // a row captured no speech at all, drop out of conversation mode
        // instead of silently polling/re-listening indefinitely.
        noSpeechStreakRef.current = hadSpeech ? 0 : noSpeechStreakRef.current + 1;
        if (noSpeechStreakRef.current >= 3) {
          conversationModeRef.current = false;
          setConversationMode(false);
          setStatusText('No speech detected — conversation ended. Tap to start again.');
        } else if (segs.length > 0) {
          // Wait for TTS playback — resumed from audioQueue's onAllDone so the
          // mic never re-arms while the kiosk's own voice is still playing.
          setStatusText('🔊 Speaking…');
        } else {
          // Nothing to play back (e.g. a silent/no-speech turn) — nothing will
          // fire onAllDone, so re-arm the mic directly after a short pause.
          setStatusText(hadSpeech ? '💬 …' : '🎧 Listening…');
          window.setTimeout(() => {
            if (conversationModeRef.current) void startRef.current?.();
          }, 300);
        }
      } else {
        setStatusText('✓ Done — tap 🎤 for another question');
      }
      // onTurnComplete also fires when TTS finishes (onAllDone); fire here too
      // in case there was no audio to play.
      if (segs.length === 0) onTurnComplete?.();
      return;
    }

    pollTimerRef.current = window.setTimeout(pollLoop, tuning.pollIntervalMs);
  }, [audioQueue, onTurnComplete, stopPolling, teardownCapture]);

  // ── Public: start recording ───────────────────────────────────────────────
  const start = useCallback(async () => {
    if (!enabled) {
      setStatusText('⏳ Ingestion in progress — please wait…');
      return;
    }
    if (recordingRef.current || phase !== 'idle') return;
    const myGen = ++startGenRef.current;
    setError(null);
    audioQueue.reset();
    framesRef.current = [];
    eosRef.current = false;
    setPartialUser('🎤 Listening…');
    setPartialAssistant('');
    setPhase('listening');
    // Do NOT invite speech yet — the mic permission, AudioContext, worklet
    // module fetch and the session POST all still have to complete. Prompting
    // here made customers start talking before capture existed.
    setStatusText('🎙 Preparing microphone…');

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Microphone access requires HTTPS or localhost.');
      }
      const constraints: MediaStreamConstraints = {
        audio: deviceId
          ? {
              // `ideal`, not `exact`: a stale/unplugged device id must fall
              // back to the default mic rather than reject the whole request
              // with OverconstrainedError and leave the kiosk mute.
              deviceId: { ideal: deviceId },
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
            }
          : { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      };
      const stream = await acquireMicWithTimeout(constraints, MIC_ACQUIRE_TIMEOUT_MS, MIC_TIMEOUT_MESSAGE);
      if (myGen !== startGenRef.current) {
        // Superseded while the permission prompt was open — release the device
        // we just acquired rather than leaking a live mic.
        stream.getTracks().forEach((t) => t.stop());
        return;
      }
      streamRef.current = stream;

      const ctx = new AudioContext();
      ctxRef.current = ctx;
      ctxRateRef.current = ctx.sampleRate;
      await ctx.audioWorklet.addModule('/pcm-capture-processor.js');
      if (myGen !== startGenRef.current) return;

      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source;
      const worklet = new AudioWorkletNode(ctx, 'pcm-capture-processor');
      workletRef.current = worklet;

      worklet.port.onmessage = (ev: MessageEvent<Float32Array>) => {
        if (!recordingRef.current) return;
        framesRef.current.push(ev.data);
        // Both modes upload continuously in small tuning.chunkSeconds (0.5s)
        // slices so the backend's per-frame VAD never falls behind while the
        // customer is still talking — withholding audio until Stop meant the
        // whole utterance's VAD/frame-processing had to run in one CPU burst
        // right after Stop, inflating post_speech_gap_ms by hundreds of ms to
        // over a second for longer utterances.
        //
        // This is safe for push-to-talk's Whisper-accuracy fix ("Aloo Tikki
        // Burger" -> "and 2,000" from 2.5s re-split chunks): startStreamSession
        // sets chunk_seconds/silence_timeout_seconds far beyond any realistic
        // utterance and disables the adaptive pre-warm flush for single-chunk
        // mode (see kioskApi.ts), so the backend only ever buffers these
        // uploads — it does not call the analyzer until the explicit
        // end-of-stream signal (Stop) is received, at which point the FULL,
        // uncut utterance is transcribed in one Whisper call exactly as
        // before. Only the upload cadence changed, not when ASR fires.
        void flushChunk(false);
      };

      source.connect(worklet);
      // Connect to destination so the worklet's process() runs; output is silent.
      worklet.connect(ctx.destination);
      // Arm capture before opening the session so nothing spoken during the
      // session round-trip is lost. flushChunk retains frames until the
      // session id exists, so this buffering is safe.
      recordingRef.current = true;

      const { session_id } = await startStreamSession(
        TARGET_RATE,
        buildHistory(),
        conversationIdRef.current,
        !conversationModeRef.current,
      );
      if (myGen !== startGenRef.current) return;
      sessionIdRef.current = session_id;

      // Everything is live — only now is it honest to ask for speech.
      setStatusText('🎙 Listening — speak now');

      // Begin the poll loop (partial transcript while listening).
      stopPolling();
      pollTimerRef.current = window.setTimeout(pollLoop, tuning.pollIntervalMs);
    } catch (err) {
      if (myGen !== startGenRef.current) return;
      recordingRef.current = false;
      teardownCapture();
      setPhase('idle');
      setPartialUser('');
      // A failed start must also drop hands-free mode. Leaving it on rendered
      // the button as a stop square while nothing was recording, so the next
      // tap read as "end conversation" instead of retrying — the button
      // appeared stuck and never sent anything.
      conversationModeRef.current = false;
      setConversationMode(false);
      const msg = describeMicError(err);
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

  // startRef must stay pointed at the latest `start` closure so the
  // conversation-mode auto-resume paths (audioQueue.onAllDone, the no-TTS
  // setTimeout above) always call a `start` that sees the current `phase`.
  startRef.current = start;

  // ── Public: hands-free conversation mode ──────────────────────────────────
  // Turns the mic button into a "start conversation" control: one tap begins
  // listening, and every subsequent turn (silence-timeout end → reply → TTS)
  // automatically re-arms the mic with no further taps, until endConversation
  // is called. The mic is fully torn down before each reply is spoken (see
  // `stop`/finalisation above) and only re-created after playback finishes,
  // so there is no acoustic echo risk from the kiosk hearing its own voice.
  const startConversation = useCallback(() => {
    conversationModeRef.current = true;
    setConversationMode(true);
    noSpeechStreakRef.current = 0;
    void start();
  }, [start]);

  const endConversation = useCallback(() => {
    conversationModeRef.current = false;
    setConversationMode(false);
    // Barge-in may have left TTS playing (see interruptSpeaking); a full exit
    // should always fall silent immediately rather than let the reply finish.
    audioQueue.stop();
    if (recordingRef.current) {
      // Finalise the in-flight utterance normally (customer may be mid-order);
      // it just won't loop again once the reply/TTS for this turn are done.
      void stop();
    } else {
      // Nothing to finalise. This branch used to be `else if (phase ===
      // 'idle')`, which silently did nothing whenever the hook was mid
      // mic-acquisition or latched in 'processing' — leaving `phase` stuck
      // non-idle so every later start() was swallowed by its own guard and the
      // button could never be used again. Always hard-reset instead.
      forceIdle();
      setStatusText('Tap the mic and ask a question');
    }
  }, [audioQueue, forceIdle, stop]);

  // ── Public: barge-in — stop the kiosk speaking and listen immediately ─────
  // Lets the customer interrupt a reply mid-playback ("stop, I want to ask
  // something else") without leaving conversation mode. This is a manual tap
  // gesture, not open-mic-during-playback: the mic stays off until this
  // fires, so there is no risk of the kiosk hearing its own voice. Silences
  // the current + any queued TTS segments, then re-arms the mic right away
  // instead of waiting for onAllDone (which audioQueue.stop() deliberately
  // does not fire, to avoid a double-resume race with this call).
  const interruptSpeaking = useCallback(() => {
    audioQueue.stop();
    if (!recordingRef.current && phase === 'idle') {
      void startRef.current?.();
    }
  }, [audioQueue, phase]);

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      conversationModeRef.current = false;
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
    conversationMode,
    start,
    stop,
    startConversation,
    endConversation,
    interruptSpeaking,
    /** Escape hatch: force the control back to a tappable idle state. */
    reset: forceIdle,
  };
}
