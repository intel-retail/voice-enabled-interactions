import type { TtsPlaybackState, VoicePhase } from '../../types';

interface MicButtonProps {
  phase: VoicePhase;
  playbackState: TtsPlaybackState;
  locked: boolean; // disabled while a knowledge-base ingest runs
  /** True once hands-free conversation mode is active (see useVoiceSession). */
  conversationMode: boolean;
  onStart: () => void;
  onStop: () => void;
  /** Silence the current reply and start listening again immediately. */
  onInterrupt: () => void;
}

/**
 * Push-to-talk mic control (default): tap to start recording, tap again (or
 * the same button, now showing a stop icon) to stop — nothing is sent to the
 * backend until that second tap. The whole utterance is uploaded as one
 * chunk on stop (see flushChunk(true)/stop() in useVoiceSession); no audio
 * streams to the backend while the button just shows "Recording…".
 *
 * Hands-free conversation mode (`conversationMode`) is a separate, currently
 * unused-by-default opt-in: one tap starts continuous listening, the kiosk
 * auto-sends after a pause in speech (backend silence-timeout VAD), speaks
 * the reply, then re-arms the mic automatically until "End conversation" is
 * pressed. Disabled (locked) while a knowledge-base ingest is in progress.
 *
 * While the kiosk is speaking in hands-free mode, a tap is treated as a
 * barge-in ("stop talking, I have another question") rather than ending the
 * conversation — it silences the reply and starts listening right away.
 */
export function MicButton({
  phase,
  playbackState,
  locked,
  conversationMode,
  onStart,
  onStop,
  onInterrupt,
}: MicButtonProps) {
  const recording = phase === 'listening';
  const processing = phase === 'processing';
  const idle = phase === 'idle';
  const speaking = playbackState !== 'idle';
  // Once conversation mode is on, the button always ends it, even while
  // "processing" between turns (not just while actively recording) — except
  // while speaking, where a tap is a barge-in instead (see onInterrupt).
  const showEnd = conversationMode && !speaking;
  const showInterrupt = conversationMode && speaking;
  const disabled = locked || (processing && !showEnd);

  const handleClick = () => {
    if (disabled) return;
    if (showInterrupt) onInterrupt();
    // Push-to-talk: a tap while actively recording must stop (and send) the
    // utterance even outside hands-free mode — `showEnd` alone only covers
    // the conversation-mode case.
    else if (showEnd || recording) onStop();
    else onStart();
  };

  // Enhanced visual states
  const base =
    'relative flex items-center justify-center w-20 h-20 rounded-full text-3xl transition-all duration-200 shadow-lg focus:outline-none focus:ring-4';
  
  const stateClass = recording
    ? 'bg-red-500 text-white kiosk-pulse-recording focus:ring-red-500/30 hover:bg-red-600'
    : showInterrupt
      ? 'bg-intel-blue text-white kiosk-pulse-recording focus:ring-intel-blue/30 hover:bg-intel-blue-dark'
      : showEnd
        ? 'bg-gray-700 text-white focus:ring-gray-500/30 hover:bg-gray-800'
        : processing
          ? 'bg-amber-500 text-white animate-spin-slow focus:ring-amber-500/30 cursor-wait'
          : disabled
            ? 'bg-gray-300 text-gray-500 cursor-not-allowed opacity-50'
            : 'bg-intel-blue text-white hover:bg-intel-blue-dark hover:scale-105 focus:ring-intel-blue/30 active:scale-95';

  // Status indicator label below button
  const statusText = locked
    ? 'Ingestion in progress...'
    : showInterrupt
      ? '🔊 Speaking… (tap to interrupt & ask)'
      : showEnd
        ? (idle ? 'Starting…' : recording ? 'Listening… (tap to end)' : 'Tap to end conversation')
        : processing
          ? 'Processing...'
          : recording
            ? 'Recording... (tap to stop)'
            : 'Tap to start recording';

  const statusColor = recording
    ? 'text-red-500'
    : showInterrupt
      ? 'text-intel-blue'
      : showEnd
        ? 'text-gray-600'
        : processing
          ? 'text-amber-500'
          : disabled
            ? 'text-gray-400'
            : 'text-intel-blue';

  return (
    <div className="flex flex-col items-center space-y-3">
      <button
        type="button"
        className={`${base} ${stateClass}`}
        onClick={handleClick}
        disabled={disabled}
        aria-pressed={recording || showEnd || showInterrupt}
        aria-label={
          showInterrupt ? 'Interrupt and ask something else' : showEnd ? 'End conversation' : recording ? 'Stop recording' : 'Start recording'
        }
        title={statusText}
      >
        {processing && !showEnd ? (
          // Processing spinner icon
          <svg className="w-8 h-8 animate-spin-slow" viewBox="0 0 24 24" fill="none">
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        ) : recording || showEnd || showInterrupt ? (
          // Stop icon for recording / hands-free conversation active / barge-in
          <svg className="w-8 h-8" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        ) : (
          // Microphone icon for idle
          <svg className="w-8 h-8" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
            <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
          </svg>
        )}

        {/* Waveform bars when speaking */}
        {recording && (
          <div className="absolute -bottom-1 flex items-end space-x-0.5 h-3">
            <div className="kiosk-bar w-0.5 bg-white rounded-full" />
            <div className="kiosk-bar w-0.5 bg-white rounded-full" />
            <div className="kiosk-bar w-0.5 bg-white rounded-full" />
            <div className="kiosk-bar w-0.5 bg-white rounded-full" />
          </div>
        )}
      </button>

      {/* Status label */}
      <div className={`text-sm font-medium ${statusColor} transition-colors duration-200`}>
        {statusText}
      </div>

      {/* Explicit full exit from hands-free mode. Kept separate from the main
          button because that button is a barge-in control while speaking —
          conflating the two would make it impossible to interrupt a reply
          without also ending the conversation. */}
      {conversationMode && (
        <button
          type="button"
          onClick={onStop}
          className="text-xs font-medium text-gray-400 underline hover:text-gray-600 focus:outline-none"
        >
          End conversation
        </button>
      )}
    </div>
  );
}

export default MicButton;
