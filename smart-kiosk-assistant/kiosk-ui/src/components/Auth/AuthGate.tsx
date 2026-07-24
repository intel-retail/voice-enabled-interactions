import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { fetchIdentityEnabled } from '../../api/identityApi';
import { LoginScreen } from './LoginScreen';
import { RegisterScreen } from './RegisterScreen';
import { AuthSuccessToast } from './AuthSuccessToast';
import type { LoyaltyProfile } from '../../types';

type GateStatus = 'checking' | 'bypass' | 'login' | 'register' | 'authenticated';

/**
 * Wraps the existing kiosk chat home page with an optional biometric auth
 * gate. The gate is driven entirely by the backend's runtime capability flag
 * (`KIOSK_CORE_IDENTITY_ENABLED`, exposed at GET /api/v1/identity/enabled):
 *   - disabled/unreachable → bypass, render children exactly as before
 *   - enabled               → require face+voice verification (or self-service
 *                              registration) before rendering children
 * This component only decides what to render; it never touches the existing
 * chat/session/ordering logic, so kiosk behaviour is unchanged when the
 * identity feature is off.
 */
export function AuthGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<GateStatus>('checking');
  const [profile, setProfile] = useState<LoyaltyProfile | null>(null);
  const [authedUserId, setAuthedUserId] = useState<string | null>(null);
  const [showSuccessToast, setShowSuccessToast] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchIdentityEnabled().then((enabled) => {
      if (cancelled) return;
      setStatus(enabled ? 'login' : 'bypass');
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const handleVerified = useCallback((p: LoyaltyProfile | null, userId: string) => {
    setProfile(p);
    setAuthedUserId(userId);
    setStatus('authenticated');
    setShowSuccessToast(true);
  }, []);

  const handleRegistered = useCallback(() => {
    // Enrolment complete — return to sign-in so the user authenticates with
    // the credentials they just registered (mirrors verify()'s own contract).
    setStatus('login');
  }, []);

  if (status === 'checking') {
    return (
      <div className="flex items-center justify-center h-full bg-gray-100">
        <p className="text-sm text-kiosk-textlo">Loading…</p>
      </div>
    );
  }

  if (status === 'bypass' || status === 'authenticated') {
    const displayName = profile?.name || authedUserId || 'Guest';
    return (
      <>
        {children}
        {showSuccessToast && (
          <AuthSuccessToast name={displayName} onDismiss={() => setShowSuccessToast(false)} />
        )}
      </>
    );
  }

  if (status === 'register') {
    return (
      <RegisterScreen onRegistered={handleRegistered} onCancel={() => setStatus('login')} />
    );
  }

  return (
    <LoginScreen onVerified={handleVerified} onRegisterRequested={() => setStatus('register')} />
  );
}

export default AuthGate;
