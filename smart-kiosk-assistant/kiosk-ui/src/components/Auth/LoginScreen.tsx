import { useCallback, useEffect, useState } from 'react';
import { useCamera } from '../../hooks/useCamera';
import { useVoiceCapture } from '../../hooks/useVoiceCapture';
import { fetchChallenge, verifyIdentity } from '../../api/identityApi';
import type { LoyaltyProfile } from '../../types';

// Hard cap only — recording stops earlier once the sentence trails into
// silence (see useVoiceCapture's voice-activity detection).
const VOICE_CLIP_MAX_SECONDS = 6;

interface LoginScreenProps {
  onVerified: (profile: LoyaltyProfile | null, userId: string) => void;
  onRegisterRequested: () => void;
}

type LoginStatus = 'idle' | 'capturing' | 'verifying' | 'error';

/**
 * Biometric login gate for the kiosk. Shows a live camera preview and an
 * on-screen challenge phrase; on "Authenticate" it captures one face frame
 * plus a short voice clip and posts both to /api/v1/identity/verify (proxied
 * through kiosk-core). On success the caller is handed the resolved loyalty
 * profile and redirected to the existing chat home page.
 */
export function LoginScreen({ onVerified, onRegisterRequested }: LoginScreenProps) {
  const camera = useCamera();
  const voice = useVoiceCapture();
  const [status, setStatus] = useState<LoginStatus>('idle');
  const [prompt, setPrompt] = useState<string>('Loading challenge phrase…');
  const [challengeId, setChallengeId] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const loadChallenge = useCallback(async () => {
    const challenge = await fetchChallenge();
    if (challenge) {
      setPrompt(challenge.prompt_text);
      setChallengeId(challenge.challenge_id);
    } else {
      setPrompt('Please look at the camera and say your name.');
      setChallengeId(null);
    }
  }, []);

  useEffect(() => {
    void camera.start();
    void loadChallenge();
    return () => camera.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleAuthenticate = useCallback(async () => {
    setErrorMsg(null);
    setStatus('capturing');
    const image_base64 = camera.captureFrameBase64();
    if (!image_base64) {
      setErrorMsg('Could not capture a camera frame. Please ensure your face is visible.');
      setStatus('error');
      return;
    }
    const audio_base64 = await voice.recordClip(VOICE_CLIP_MAX_SECONDS);
    if (!audio_base64) {
      setErrorMsg(voice.error ?? 'Could not capture audio.');
      setStatus('error');
      return;
    }
    setStatus('verifying');
    try {
      const result = await verifyIdentity({
        challenge_id: challengeId,
        image_base64,
        audio_base64,
      });
      if (result.verified && result.user_id) {
        onVerified(result.profile ?? null, result.user_id);
      } else {
        setErrorMsg(result.reason ?? 'User not authenticated. Please try again or register.');
        setStatus('error');
        void loadChallenge();
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMsg(`User not authenticated: ${msg}`);
      setStatus('error');
    }
  }, [camera, voice, challengeId, onVerified, loadChallenge]);

  const busy = status === 'capturing' || status === 'verifying';

  return (
    <div className="flex flex-col items-center justify-center h-full bg-gray-100 px-4 py-8 gap-6">
      <div className="max-w-2xl w-full bg-white rounded-xl border border-gray-200 shadow-sm p-8 sm:p-10 flex flex-col items-center gap-5">
        <h1 className="text-2xl font-semibold text-intel-blue">Sign in to the kiosk</h1>
        <p className="text-base text-kiosk-textlo text-center">
          Look at the camera and read the phrase below aloud.
        </p>

        <div className="w-full max-w-lg aspect-video bg-black rounded-lg overflow-hidden">
          <video ref={camera.videoRef} muted playsInline className="w-full h-full object-cover" />
        </div>

        <div className="w-full bg-gray-50 border border-gray-200 rounded-lg px-6 py-4 text-center">
          <span className="text-xs uppercase tracking-wide text-kiosk-textlo">Say aloud</span>
          <p className="text-lg font-medium text-gray-800">{prompt}</p>
        </div>

        {camera.error && <p className="text-sm text-red-500 text-center">{camera.error}</p>}

        {status === 'error' && errorMsg && (
          <div className="w-full bg-red-50 border border-red-200 text-red-700 text-sm rounded-lg px-4 py-3 text-center">
            {errorMsg}
          </div>
        )}

        <button
          type="button"
          onClick={handleAuthenticate}
          disabled={busy || !camera.ready}
          className="w-full max-w-sm rounded-full bg-intel-blue text-white font-medium text-lg py-4 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-intel-blue-dark transition-colors"
        >
          {status === 'capturing'
            ? 'Capturing…'
            : status === 'verifying'
              ? 'Verifying…'
              : 'Authenticate'}
        </button>

        <button
          type="button"
          onClick={onRegisterRequested}
          disabled={busy}
          className="text-base text-intel-blue underline disabled:opacity-50"
        >
          New here? Register your face &amp; voice
        </button>
      </div>
    </div>
  );
}

export default LoginScreen;
