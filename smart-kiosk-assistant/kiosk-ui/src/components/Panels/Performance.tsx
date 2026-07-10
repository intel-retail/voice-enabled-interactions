import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { toChartPoints } from '../../api/metricsApi';
import { useMetrics } from '../../hooks/useMetrics';
import type { ChartPoint } from '../../types';

interface MiniChartProps {
  title: string;
  points: ChartPoint[];
  color: string;
}

function MiniChart({ title, points, color }: MiniChartProps) {
  return (
    <section>
      <h3 className="text-xs font-medium text-intel-dark text-center mb-1">{title}</h3>
      {points.length === 0 ? (
        <div className="flex h-[150px] items-center justify-center text-xs text-kiosk-textlo">
          Waiting for data…
        </div>
      ) : (
        <div className="h-[150px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={points}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="time" tick={{ fontSize: 9 }} minTickGap={24} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 9 }} width={32} unit="%" />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="value"
                stroke={color}
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </section>
  );
}

export default function Performance() {
  const { metrics, refresh } = useMetrics();
  const cpu = toChartPoints(metrics.cpu_utilization);
  const gpu = toChartPoints(metrics.gpu_utilization);
  const npu = toChartPoints(metrics.npu_utilization);
  const mem = toChartPoints(metrics.memory, 4);

  return (
    <div className="space-y-3">
      <MiniChart title="CPU" points={cpu} color="#3aa0eb" />
      <MiniChart title="GPU" points={gpu} color="#4bc0c0" />
      <MiniChart title="Memory" points={mem} color="#ff6384" />
      <MiniChart title="NPU" points={npu} color="#9966ff" />
      <div className="flex justify-center">
        <button
          type="button"
          className="text-sm px-3 py-1.5 rounded-md border border-kiosk-border text-intel-dark hover:bg-kiosk-pane"
          onClick={refresh}
        >
          🔄 Refresh
        </button>
      </div>
    </div>
  );
}
