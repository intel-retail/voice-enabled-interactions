import { useCallback, useEffect, useRef, useState } from 'react';
import type { TtsPlaybackState } from '../types';

/**
 * Sequential TTS audio playback queue.
 *
 * The voice session enqueues TTS segment URLs as they become available; this
 * hook plays them one after another via a single HTMLAudioElement, tracks a
 * playback state for the assistant indicator, and de-dupes already-played URLs.
 *
 * Autoplay is permitted because playback begins after the user's mic-tap gesture.
 */
export function useAudioQueue(options?: {
  onFirstPlay?: () => void;
  onAllDone?: () => void;
}) {
  const [state, setState] = useState<TtsPlaybackState>('idle');
  const queueRef = useRef<string[]>([]);
  const playedRef = useRef<Set<string>>(new Set());
  const playingRef = useRef(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const firstPlayFiredRef = useRef(false);
  const optsRef = useRef(options);
  optsRef.current = options;

  // Lazily create the shared audio element.
  const ensureAudio = useCallback((): HTMLAudioElement => {
    if (!audioRef.current) {
      audioRef.current = new Audio();
      audioRef.current.preload = 'auto';
    }
    return audioRef.current;
  }, []);

  const playNext = useCallback(() => {
    const next = queueRef.current.shift();
    if (!next) {
      playingRef.current = false;
      setState('idle');
      if (firstPlayFiredRef.current) {
        firstPlayFiredRef.current = false;
        optsRef.current?.onAllDone?.();
      }
      return;
    }

    playingRef.current = true;
    setState('playing');
    const audio = ensureAudio();
    audio.src = next;

    if (!firstPlayFiredRef.current) {
      firstPlayFiredRef.current = true;
      optsRef.current?.onFirstPlay?.();
    }

    const onEnded = () => {
      audio.removeEventListener('ended', onEnded);
      audio.removeEventListener('error', onError);
      playNext();
    };
    const onError = () => {
      audio.removeEventListener('ended', onEnded);
      audio.removeEventListener('error', onError);
      // Skip the failed segment and continue.
      playNext();
    };
    audio.addEventListener('ended', onEnded);
    audio.addEventListener('error', onError);

    audio.play().catch(() => {
      // Autoplay blocked or load error — skip to the next.
      audio.removeEventListener('ended', onEnded);
      audio.removeEventListener('error', onError);
      playNext();
    });
  }, [ensureAudio]);

  /** Enqueue new TTS URLs (already-played URLs are ignored). */
  const enqueue = useCallback(
    (urls: string[]) => {
      let added = false;
      for (const url of urls) {
        if (!url || playedRef.current.has(url)) continue;
        playedRef.current.add(url);
        queueRef.current.push(url);
        added = true;
      }
      if (added && !playingRef.current) {
        setState('queued');
        playNext();
      }
    },
    [playNext],
  );

  /** Reset the queue for a new conversation turn (does not stop current audio). */
  const reset = useCallback(() => {
    queueRef.current = [];
    playedRef.current = new Set();
  }, []);

  /** Hard stop: clear queue and stop any playing audio. */
  const stop = useCallback(() => {
    queueRef.current = [];
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.src = '';
    }
    playingRef.current = false;
    firstPlayFiredRef.current = false;
    setState('idle');
  }, []);

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause();
        audioRef.current.src = '';
      }
    };
  }, []);

  return { state, enqueue, reset, stop };
}
