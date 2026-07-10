import { useCallback, useEffect, useState } from 'react';

interface UseMicDevicesResult {
  devices: MediaDeviceInfo[];
  selectedId: string;
  setSelectedId: (id: string) => void;
  refresh: () => Promise<void>;
  error: string | null;
}

const unsupportedMessage = 'Microphone access requires HTTPS or localhost.';
const permissionMessage = 'Unable to access the microphone. Device names may be unavailable until permission is granted.';
const enumerateMessage = 'Unable to list microphones. Check your browser permissions.';

export default function useMicDevices(): UseMicDevicesResult {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedId, setSelectedId] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async (): Promise<void> => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setDevices([]);
      setSelectedId('');
      setError(unsupportedMessage);
      return;
    }

    let permissionError: string | null = null;

    if (navigator.mediaDevices.getUserMedia) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        stream.getTracks().forEach((track) => track.stop());
      } catch {
        permissionError = permissionMessage;
      }
    }

    try {
      const nextDevices = (await navigator.mediaDevices.enumerateDevices()).filter(
        (device) => device.kind === 'audioinput',
      );

      setDevices(nextDevices);
      setSelectedId((currentId) => {
        if (currentId && nextDevices.some((device) => device.deviceId === currentId)) {
          return currentId;
        }

        return nextDevices[0]?.deviceId ?? '';
      });
      setError(permissionError);
    } catch {
      setDevices([]);
      setSelectedId('');
      setError(permissionError ?? enumerateMessage);
    }
  }, []);

  useEffect(() => {
    void refresh();

    if (!navigator.mediaDevices?.addEventListener) {
      return undefined;
    }

    navigator.mediaDevices.addEventListener('devicechange', refresh);

    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', refresh);
    };
  }, [refresh]);

  return { devices, selectedId, setSelectedId, refresh, error };
}
