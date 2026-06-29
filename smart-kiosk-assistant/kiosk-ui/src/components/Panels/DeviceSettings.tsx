interface DeviceSettingsProps {
  devices: MediaDeviceInfo[];
  selectedId: string;
  onSelect: (id: string) => void;
  error?: string | null;
}

export default function DeviceSettings({
  devices,
  selectedId,
  onSelect,
  error,
}: DeviceSettingsProps) {
  return (
    <div>
      <p className="text-sm text-kiosk-textmd mb-2">Select the microphone to use for recording.</p>
      <select
        className="w-full border border-kiosk-border rounded-md px-3 py-2 text-sm text-intel-dark bg-white focus:outline-none focus:ring-2 focus:ring-intel-blue"
        value={selectedId}
        onChange={(event) => onSelect(event.target.value)}
      >
        {devices.map((device, index) => (
          <option key={device.deviceId} value={device.deviceId}>
            {device.label || `Microphone ${index + 1}`}
          </option>
        ))}
      </select>
      {error ? <p className="text-xs text-amber-600 mt-2">{error}</p> : null}
      {devices.length === 0 ? (
        <p className="text-xs text-kiosk-textlo mt-2">No microphones detected.</p>
      ) : null}
    </div>
  );
}
