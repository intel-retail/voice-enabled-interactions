import { endpoints } from '../constants';
import type {
  ChallengeResponse,
  RegisterRequest,
  RegisterResponse,
  VerifyRequest,
  VerifyResponse,
} from '../types';

/**
 * Runtime capability probe — mirrors the backend's KIOSK_CORE_IDENTITY_ENABLED
 * flag. This endpoint is always reachable (unlike the flag-gated identity
 * router) so the UI can decide gate-vs-bypass without a rebuild. Any failure
 * (network error, service down) is treated as "disabled" so the kiosk falls
 * back to the existing chat behaviour rather than blocking the user.
 */
export async function fetchIdentityEnabled(): Promise<boolean> {
  try {
    const res = await fetch(endpoints.identityEnabled, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return false;
    const data: { enabled: boolean } = await res.json();
    return Boolean(data.enabled);
  } catch {
    return false;
  }
}

/** Fetch a random anti-replay voice challenge prompt for the user to read aloud. */
export async function fetchChallenge(): Promise<ChallengeResponse | null> {
  try {
    const res = await fetch(endpoints.identityChallenge, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

/** Verify a captured face frame + voice clip against enrolled loyalty profiles. */
export async function verifyIdentity(request: VerifyRequest): Promise<VerifyResponse> {
  const res = await fetch(endpoints.identityVerify, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal: AbortSignal.timeout(15000),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`Verification request failed (${res.status}): ${detail}`);
  }
  return await res.json();
}

/** Self-service enrolment: register a new loyalty profile from face + voice. */
export async function registerIdentity(request: RegisterRequest): Promise<RegisterResponse> {
  const res = await fetch(endpoints.identityRegister, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
    signal: AbortSignal.timeout(15000),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`Registration request failed (${res.status}): ${detail}`);
  }
  return await res.json();
}
