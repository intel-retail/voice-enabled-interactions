import { useCallback, useEffect, useRef, useState } from 'react';
import type { TtsPlaybackState } from '../types';

/**
 * Gapless sequential TTS audio playback queue.
 *
 * The assistant's reply is synthesised as one file per clause so playback can
 * start as soon as the first fragment is ready (measured time-to-first-audio is
 * ~387ms this way versus ~1769ms when synthesising whole sentences). The cost of
 * that split is that a naive player pays a fetch/decode/start penalty at every
 * fragment boundary, which is heard as a stall at each comma.
 *
 * This hook removes that penalty by decoding segments ahead of time and
 * scheduling them on a single AudioContext timeline: each buffer is started at
 * the exact moment its predecessor ends, so the reply plays as one continuous
 * utterance. Decoding happens concurrently with playback, and segments are
 * always scheduled in arrival order.
 *
 * Autoplay is permitted because playback begins after the user's mic-tap gesture.
 */
export function useAudioQueue(options?: {
  onFirstPlay?: () => void;
  onAllDone?: () => void;
}) {
  const [state, setState] = useState<TtsPlaybackState>('idle');

  const ctxRef = useRef<AudioContext | null>(null);
  const playedRef = useRef<Set<string>>(new Set());
  // Segments awaiting scheduling, in arrival order. Decoding starts on enqueue,
  // so by the time a segment reaches the head its buffer is usually ready.
  const pendingRef = useRef<Array<Promise<AudioBuffer | null>>>([]);
  const activeSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
  // Timeline cursor: when the next segment should begin.
  const nextStartRef = useRef(0);
  const pumpingRef = useRef(false);
  const scheduledRef = useRef(0);
  const finishedRef = useRef(0);
  const firstPlayFiredRef = useRef(false);
  const generationRef = useRef(0);

  const optsRef = useRef(options);
  optsRef.current = options;

  const ensureCtx = useCallback((): AudioContext => {
    if (!ctxRef.current || ctxRef.current.state === 'closed') {
      const Ctor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      ctxRef.current = new Ctor();
      nextStartRef.current = 0;
    }
    return ctxRef.current;
  }, []);

  /** Fire onAllDone once every scheduled segment has finished playing. */
  const settle = useCallback(() => {
    if (pendingRef.current.length > 0) return;
    if (finishedRef.current < scheduledRef.current) return;
    setState('idle');
    if (firstPlayFiredRef.current) {
      firstPlayFiredRef.current = false;
      optsRef.current?.onAllDone?.();
    }
  }, []);

  /**
   * Drain the pending queue, scheduling each decoded buffer immediately after
   * its predecessor. Runs as a single logical loop so ordering is preserved
   * even though decoding is concurrent.
   */
  const pump = useCallback(async () => {
    if (pumpingRef.current) return;
    pumpingRef.current = true;
    const generation = generationRef.current;

    try {
      while (pendingRef.current.length > 0) {
        const head = pendingRef.current[0];
        const buffer = await head;
        // A reset/stop happened while we were decoding — abandon this run.
        if (generation !== generationRef.current) return;
        pendingRef.current.shift();
        if (!buffer) continue;

        const ctx = ensureCtx();
        if (ctx.state === 'suspended') {
          try {
            await ctx.resume();
          } catch {
            /* resume rejects only when the gesture was lost; play anyway */
          }
          if (generation !== generationRef.current) return;
        }

        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(ctx.destination);

        // Never schedule in the past: if decoding fell behind playback the
        // cursor is stale, so restart it from now (a gap is unavoidable there).
        const startAt = Math.max(ctx.currentTime, nextStartRef.current);
        source.start(startAt);
        nextStartRef.current = startAt + buffer.duration;

        scheduledRef.current += 1;
        activeSourcesRef.current.add(source);
        source.onended = () => {
          activeSourcesRef.current.delete(source);
          if (generation !== generationRef.current) return;
          finishedRef.current += 1;
          settle();
        };

        setState('playing');
        if (!firstPlayFiredRef.current) {
          firstPlayFiredRef.current = true;
          optsRef.current?.onFirstPlay?.();
        }
      }
    } finally {
      pumpingRef.current = false;
    }

    if (generation === generationRef.current) settle();
  }, [ensureCtx, settle]);

  /** Enqueue new TTS URLs (already-played URLs are ignored). */
  const enqueue = useCallback(
    (urls: string[]) => {
      let added = false;
      for (const url of urls) {
        if (!url || playedRef.current.has(url)) continue;
        playedRef.current.add(url);

        // Start fetching and decoding straight away so the work overlaps with
        // playback of earlier segments; a failed segment resolves to null and
        // is skipped rather than stalling the reply.
        const ctx = ensureCtx();
        pendingRef.current.push(
          fetch(url)
            .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(String(r.status)))))
            .then((buf) => ctx.decodeAudioData(buf))
            .catch(() => null),
        );
        added = true;
      }
      if (!added) return;
      setState((prev) => (prev === 'playing' ? prev : 'queued'));
      void pump();
    },
    [ensureCtx, pump],
  );

  /** Reset the queue for a new conversation turn (does not stop current audio). */
  const reset = useCallback(() => {
    pendingRef.current = [];
    playedRef.current = new Set();
  }, []);

  /** Hard stop: drop pending work and silence anything already scheduled. */
  const stop = useCallback(() => {
    generationRef.current += 1;
    pendingRef.current = [];
    for (const source of activeSourcesRef.current) {
      try {
        source.stop();
      } catch {
        /* already stopped */
      }
    }
    activeSourcesRef.current.clear();
    scheduledRef.current = 0;
    finishedRef.current = 0;
    nextStartRef.current = ctxRef.current ? ctxRef.current.currentTime : 0;
    firstPlayFiredRef.current = false;
    setState('idle');
  }, []);

  useEffect(() => {
    return () => {
      generationRef.current += 1;
      for (const source of activeSourcesRef.current) {
        try {
          source.stop();
        } catch {
          /* already stopped */
        }
      }
      activeSourcesRef.current.clear();
      if (ctxRef.current && ctxRef.current.state !== 'closed') void ctxRef.current.close();
      ctxRef.current = null;
    };
  }, []);

  return { state, enqueue, reset, stop };
}
