import { useCallback, useEffect, useState } from 'react';
import { useCamera } from '../../hooks/useCamera';
import { useVoiceCapture } from '../../hooks/useVoiceCapture';
import { fetchChallenge, registerIdentity } from '../../api/identityApi';

// Hard cap only — recording stops earlier once the sentence trails into
// silence (see useVoiceCapture's voice-activity detection).
const VOICE_CLIP_MAX_SECONDS = 6;

interface RegisterScreenProps {
  onRegistered: (userId: string) => void;
  onCancel: () => void;
}

type RegisterStatus = 'idle' | 'capturing' | 'submitting' | 'error' | 'success';

/** Slugify a display name and append a short random suffix for uniqueness. */
function generateUserId(name: string): string {
  const slug = name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)+/g, '') || 'guest';
  const suffix = Math.random().toString(36).slice(2, 7);
  return `${slug}-${suffix}`;
}

/**
 * Self-service enrolment screen. Collects a display name, shows an on-screen
 * challenge phrase, captures one face frame + a short voice clip, and posts
 * both to /api/v1/identity/register (proxied through kiosk-core). This is a
 * pure add-on to the existing bootstrap/video-file enrolment path used by
 * identity-service — it reuses the same register() pipeline.
 */
export function RegisterScreen({ onRegistered, onCancel }: RegisterScreenProps) {
  const camera = useCamera();
  const voice = useVoiceCapture();
  const [name, setName] = useState('');
  const [status, setStatus] = useState<RegisterStatus>('idle');
  const [prompt, setPrompt] = useState<string>('Loading challenge phrase…');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const loadChallenge = useCallback(async () => {
    const challenge = await fetchChallenge();
    setPrompt(challenge?.prompt_text ?? 'Please say your name clearly.');
  }, []);

  useEffect(() => {
    void camera.start();
    void loadChallenge();
    return () => camera.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRegister = useCallback(async () => {
    setErrorMsg(null);
    if (!name.trim()) {
      setErrorMsg('Please enter your name first.');
      setStatus('error');
      return;
    }
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
    setStatus('submitting');
    const userId = generateUserId(name);
    try {
      const result = await registerIdentity({
        user_id: userId,
        name: name.trim(),
        image_base64,
        audio_base64,
      });
      if (result.registered) {
        setStatus('success');
        onRegistered(result.user_id);
      } else {
        setErrorMsg(result.reason ?? 'Registration failed. Please try again.');
        setStatus('error');
        void loadChallenge();
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMsg(`Registration failed: ${msg}`);
      setStatus('error');
    }
  }, [camera, voice, name, onRegistered, loadChallenge]);

  const busy = status === 'capturing' || status === 'submitting';

  return (
    <div className="flex flex-col items-center justify-center h-full bg-gray-100 px-4 py-8 gap-6">
      <div className="max-w-2xl w-full bg-white rounded-xl border border-gray-200 shadow-sm p-8 sm:p-10 flex flex-col items-center gap-5">
        <h1 className="text-2xl font-semibold text-intel-blue">Register your face &amp; voice</h1>
        <p className="text-base text-kiosk-textlo text-center">
          Enter your name, then read the phrase below aloud while looking at the camera.
        </p>

        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Your name"
          disabled={busy}
          className="w-full max-w-lg rounded-lg border border-gray-300 px-4 py-3 text-base focus:outline-none focus:ring-2 focus:ring-intel-blue/30"
        />

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
          onClick={handleRegister}
          disabled={busy || !camera.ready}
          className="w-full max-w-sm rounded-full bg-intel-blue text-white font-medium text-lg py-4 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-intel-blue-dark transition-colors"
        >
          {status === 'capturing'
            ? 'Capturing…'
            : status === 'submitting'
              ? 'Registering…'
              : 'Register'}
        </button>

        <button
          type="button"
          onClick={onCancel}
          disabled={busy}
          className="text-base text-intel-blue underline disabled:opacity-50"
        >
          Back to sign in
        </button>
      </div>
    </div>
  );
}

export default RegisterScreen;
