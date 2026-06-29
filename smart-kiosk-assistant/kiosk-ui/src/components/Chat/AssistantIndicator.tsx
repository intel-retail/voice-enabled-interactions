import type { TtsPlaybackState, VoicePhase } from '../../types';

interface AssistantIndicatorProps {
  phase: VoicePhase;
  playbackState: TtsPlaybackState;
}

/**
 * Animated status indicator that mirrors the assistant's current activity:
 * listening, thinking, or speaking (animated bars during TTS playback).
 */
export function AssistantIndicator({ phase, playbackState }: AssistantIndicatorProps) {
  const speaking = playbackState === 'playing' || playbackState === 'queued';

  let label = '';
  let icon = null;
  let colorClass = 'text-intel-blue';

  if (speaking) {
    label = 'Assistant speaking...';
    icon = (
      <div className="flex items-end gap-0.5 h-5" aria-hidden>
        <span className="kiosk-bar w-1 h-5 bg-intel-blue rounded-sm" />
        <span className="kiosk-bar w-1 h-5 bg-intel-blue rounded-sm" />
        <span className="kiosk-bar w-1 h-5 bg-intel-blue rounded-sm" />
        <span className="kiosk-bar w-1 h-5 bg-intel-blue rounded-sm" />
      </div>
    );
  } else if (phase === 'processing') {
    label = 'Thinking...';
    colorClass = 'text-amber-500';
    icon = (
      <svg className="w-4 h-4 animate-spin-slow" viewBox="0 0 24 24" fill="none">
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
    );
  } else if (phase === 'listening') {
    label = 'Listening...';
    colorClass = 'text-red-500';
    icon = (
      <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
        <circle cx="12" cy="12" r="3" className="animate-pulse" />
        <circle cx="12" cy="12" r="8" opacity="0.3" />
      </svg>
    );
  } else {
    return null;
  }

  return (
    <div
      className={`inline-flex items-center gap-2 px-3 py-2 rounded-full bg-white border border-gray-200 shadow-sm ${colorClass} transition-all duration-200`}
    >
      {icon}
      <span className="text-xs font-medium">{label}</span>
    </div>
  );
}

export default AssistantIndicator;
