import type { VoicePhase } from '../../types';

interface MicButtonProps {
  phase: VoicePhase;
  locked: boolean; // disabled while a knowledge-base ingest runs
  onStart: () => void;
  onStop: () => void;
}

/**
 * Press-to-record microphone button. Tap to start recording, tap again to stop
 * and submit. Disabled (locked) while the assistant is processing or an ingest
 * is in progress.
 */
export function MicButton({ phase, locked, onStart, onStop }: MicButtonProps) {
  const recording = phase === 'listening';
  const processing = phase === 'processing';
  const disabled = locked || processing;

  const handleClick = () => {
    if (disabled) return;
    if (recording) onStop();
    else onStart();
  };

  // Enhanced visual states
  const base =
    'relative flex items-center justify-center w-20 h-20 rounded-full text-3xl transition-all duration-200 shadow-lg focus:outline-none focus:ring-4';
  
  const stateClass = recording
    ? 'bg-red-500 text-white kiosk-pulse-recording focus:ring-red-500/30 hover:bg-red-600'
    : processing
      ? 'bg-amber-500 text-white animate-spin-slow focus:ring-amber-500/30 cursor-wait'
      : disabled
        ? 'bg-gray-300 text-gray-500 cursor-not-allowed opacity-50'
        : 'bg-intel-blue text-white hover:bg-intel-blue-dark hover:scale-105 focus:ring-intel-blue/30 active:scale-95';

  // Status indicator label below button
  const statusText = locked
    ? 'Ingestion in progress...'
    : processing
      ? 'Processing...'
      : recording
        ? 'Recording... (tap to stop)'
        : 'Tap to speak';

  const statusColor = recording
    ? 'text-red-500'
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
        aria-pressed={recording}
        aria-label={recording ? 'Stop recording' : 'Start recording'}
        title={statusText}
      >
        {processing ? (
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
        ) : recording ? (
          // Stop icon for recording
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
    </div>
  );
}

export default MicButton;
