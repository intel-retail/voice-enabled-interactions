import { useEffect, useState } from 'react';

interface AuthSuccessToastProps {
  name: string;
  onDismiss: () => void;
  durationMs?: number;
}

/**
 * Brief overlay toast confirming a successful biometric login, shown on top of
 * the existing chat home page. Auto-dismisses after `durationMs`.
 */
export function AuthSuccessToast({ name, onDismiss, durationMs = 3000 }: AuthSuccessToastProps) {
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const hideTimer = window.setTimeout(() => setVisible(false), durationMs);
    const dismissTimer = window.setTimeout(onDismiss, durationMs + 300);
    return () => {
      window.clearTimeout(hideTimer);
      window.clearTimeout(dismissTimer);
    };
  }, [durationMs, onDismiss]);

  return (
    <div
      className={`fixed top-20 left-1/2 -translate-x-1/2 z-[100] transition-all duration-300 ${
        visible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-2 pointer-events-none'
      }`}
      role="status"
    >
      <div className="flex items-center gap-3 bg-white border border-green-200 shadow-lg rounded-xl px-5 py-3">
        <div className="flex items-center justify-center w-8 h-8 rounded-full bg-green-100 text-green-600 shrink-0">
          <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none">
            <path
              d="M20 6L9 17l-5-5"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <div>
          <p className="text-sm font-semibold text-gray-800">Successfully authenticated</p>
          <p className="text-xs text-kiosk-textlo">Welcome, {name}!</p>
        </div>
      </div>
    </div>
  );
}

export default AuthSuccessToast;
