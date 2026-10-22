import { buildStreamSocketUrl, buildStreamStartPayload } from './kioskApi';
import type { HistoryTurn } from '../types';

/**
 * Persistent hands-free uplink to kiosk-core.
 *
 * Replaces the per-chunk REST POSTs (useVoiceSession.flushChunk) for
 * conversation mode only: one full-duplex WebSocket carries a JSON `start`
 * message, then bare binary PCM16 frames, then control events (`playback_started`,
 * `end`, `cancel`). Results are still polled over REST, so only the uplink and
 * the playback report move onto the socket.
 */
export interface VoiceSocketOptions {
  /** Fired once kiosk-core acknowledges the session with its id. */
  onStarted: (sessionId: string) => void;
  /** Fired on socket error or unexpected close before `end`. */
  onError?: (message: string) => void;
}

export class VoiceStreamSocket {
  private ws: WebSocket;
  private started = false;
  private closedByUs = false;
  /** PCM frames captured before `started` arrives, flushed on ack. */
  private pending: ArrayBuffer[] = [];
  private readonly opts: VoiceSocketOptions;

  constructor(
    sampleRate: number,
    history: HistoryTurn[],
    conversationId: string | undefined,
    opts: VoiceSocketOptions,
  ) {
    this.opts = opts;
    this.ws = new WebSocket(buildStreamSocketUrl());
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.ws.send(
        JSON.stringify({
          type: 'start',
          ...buildStreamStartPayload(sampleRate, history, conversationId),
        }),
      );
    };

    this.ws.onmessage = (ev: MessageEvent) => {
      if (typeof ev.data !== 'string') return;
      let msg: { type?: string; session_id?: string; detail?: string };
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === 'started' && msg.session_id) {
        this.started = true;
        this.opts.onStarted(msg.session_id);
        // Flush any audio captured during the open/start round-trip so the
        // opening words are never lost.
        for (const frame of this.pending) this.ws.send(frame);
        this.pending = [];
      } else if (msg.type === 'error') {
        this.opts.onError?.(msg.detail ?? 'stream error');
      }
    };

    this.ws.onerror = () => {
      if (!this.closedByUs) this.opts.onError?.('websocket error');
    };
    this.ws.onclose = () => {
      if (!this.closedByUs && !this.started) {
        this.opts.onError?.('websocket closed before the session started');
      }
    };
  }

  /** Queue or send one raw PCM16 frame. */
  sendPcm(frame: ArrayBuffer): void {
    if (this.ws.readyState !== WebSocket.OPEN) return;
    if (!this.started) {
      this.pending.push(frame);
      return;
    }
    this.ws.send(frame);
  }

  /** Report that the first answer segment began playing (stops the v2v clock). */
  playbackStarted(): void {
    this.sendControl('playback_started');
  }

  /** Signal end-of-turn so kiosk-core finalises (RAG + TTS). */
  end(): void {
    this.sendControl('end');
  }

  private sendControl(type: string): void {
    if (this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type }));
    }
  }

  /** Close the socket (also ends the turn server-side via its disconnect path). */
  close(): void {
    this.closedByUs = true;
    try {
      this.ws.close();
    } catch {
      /* already closing */
    }
  }
}
