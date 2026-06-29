/**
 * HardwareCharts — real-time 60-second line charts for CPU / GPU / NPU / Memory.
 *
 * Color coding:  CPU = Intel Blue  |  GPU = Green  |  NPU = Purple  |  MEM = Slate
 *
 * Each chart shows:
 *   - Current utilization % in large text (top-right)
 *   - Device tag (CPU / GPU / NPU)
 *   - Filled area chart over 60-second window
 *   - Horizontal reference lines at 50% and 90%
 */

import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import type { ChartPoint, MetricsResponse } from '../../types';
import { toChartPoints } from '../../api/metricsApi';

// Trim to the last N points (60-second window at ~10s poll = 6 pts, more if faster)
function last60s(points: ChartPoint[]): ChartPoint[] {
  return points.slice(-30);
}

function currentValue(points: ChartPoint[]): number | null {
  if (points.length === 0) return null;
  return points[points.length - 1].value;
}

function utilColor(pct: number | null): string {
  if (pct === null) return 'text-gray-400';
  if (pct >= 90) return 'text-red-400';
  if (pct >= 70) return 'text-amber-400';
  return 'text-green-400';
}

interface HardwareChartProps {
  label: string;
  deviceTag: string;
  points: ChartPoint[];
  stroke: string;       // CSS hex
  fill: string;         // CSS hex (lighter)
  fillOpacity?: number;
}

function HardwareChart({ label, deviceTag, points, stroke, fill, fillOpacity = 0.15 }: HardwareChartProps) {
  const trimmed = last60s(points);
  const current = currentValue(trimmed);

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-3">
      {/* Chart header */}
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className="inline-block h-2.5 w-2.5 rounded-sm"
            style={{ backgroundColor: stroke }}
          />
          <span className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">
            {label}
          </span>
          <span
            className="rounded border px-1.5 py-0 text-[9px] font-bold"
            style={{
              color: stroke,
              borderColor: stroke + '50',
              backgroundColor: stroke + '15',
            }}
          >
            {deviceTag}
          </span>
        </div>

        {/* Current value */}
        <div className="text-right">
          {current !== null ? (
            <span
              className={`font-mono text-xl font-bold ${utilColor(current)}`}
              key={String(Math.round(current))}
              style={{ animation: 'number-tick 0.25s ease-out' }}
            >
              {Math.round(current)}
              <span className="ml-0.5 text-xs font-normal opacity-60">%</span>
            </span>
          ) : (
            <span className="text-xs text-gray-400">–</span>
          )}
        </div>
      </div>

      {/* Chart body */}
      {trimmed.length === 0 ? (
        <div className="flex h-[90px] items-center justify-center">
          <div className="flex flex-col items-center gap-1">
            <div className="h-4 w-4 animate-spin-slow rounded-full border-2 border-gray-200 border-t-transparent" />
            <span className="text-[10px] text-gray-400">Waiting for data…</span>
          </div>
        </div>
      ) : (
        <div className="h-[90px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={trimmed} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id={`grad-${label}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={fill}  stopOpacity={fillOpacity * 3} />
                  <stop offset="95%" stopColor={fill}  stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#e2e8f0"
                vertical={false}
              />
              <ReferenceLine y={90} stroke="#ef444460" strokeDasharray="4 2" strokeWidth={1} />
              <ReferenceLine y={50} stroke="#33415540" strokeDasharray="4 2" strokeWidth={1} />
              <XAxis dataKey="time" hide />
              <YAxis
                domain={[0, 100]}
                tick={{ fontSize: 9, fill: '#64748b' }}
                width={26}
                unit="%"
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                contentStyle={{
                  background: '#ffffff',
                  border: '1px solid #e2e8f0',
                  borderRadius: '6px',
                  fontSize: '11px',
                  color: '#1e293b',
                }}
                itemStyle={{ color: stroke }}
                formatter={(val: number) => [`${val.toFixed(1)}%`, label]}
                labelStyle={{ color: '#94a3b8', marginBottom: '2px' }}
              />
              <Area
                type="monotone"
                dataKey="value"
                stroke={stroke}
                strokeWidth={2}
                fill={`url(#grad-${label})`}
                dot={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

interface HardwareChartsProps {
  metrics: MetricsResponse;
}

export function HardwareCharts({ metrics }: HardwareChartsProps) {
  const cpu = toChartPoints(metrics.cpu_utilization);
  const gpu = toChartPoints(metrics.gpu_utilization);
  const npu = toChartPoints(metrics.npu_utilization);
  const mem = toChartPoints(metrics.memory, 4);

  const cpuNow = currentValue(last60s(cpu));
  const gpuNow = currentValue(last60s(gpu));
  const npuNow = currentValue(last60s(npu));

  return (
    <div className="space-y-2">
      {/* Section header + live badge row */}
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-400">
          Hardware Utilization
        </h2>
        <span className="flex items-center gap-1.5 rounded-full bg-green-100 px-2.5 py-0.5 text-[10px] font-semibold text-green-700">
          <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-green-500" />
          LIVE
        </span>
      </div>

      {/* Summary bar: CPU / GPU / NPU at a glance */}
      <div className="grid grid-cols-3 gap-2">
        {[
          { label: 'CPU', val: cpuNow, color: '#0071c5' },
          { label: 'GPU', val: gpuNow, color: '#16a34a' },
          { label: 'NPU', val: npuNow, color: '#9333ea' },
        ].map(({ label, val, color }) => (
          <div
            key={label}
            className="flex flex-col items-center rounded-lg border border-gray-200 bg-white py-2"
          >
            <span className="text-[10px] font-semibold uppercase tracking-wider text-gray-400">
              {label}
            </span>
            <span
              className="mt-0.5 font-mono text-2xl font-bold"
              style={{ color }}
            >
              {val !== null ? `${Math.round(val)}` : '—'}
            </span>
            <span className="text-[9px] text-gray-400">%</span>
          </div>
        ))}
      </div>

      {/* Full charts */}
      <HardwareChart
        label="CPU Utilization"
        deviceTag="CPU"
        points={cpu}
        stroke="#0071c5"
        fill="#0071c5"
      />
      <HardwareChart
        label="GPU Utilization"
        deviceTag="GPU"
        points={gpu}
        stroke="#16a34a"
        fill="#16a34a"
      />
      <HardwareChart
        label="NPU Utilization"
        deviceTag="NPU"
        points={npu}
        stroke="#9333ea"
        fill="#9333ea"
      />
      <HardwareChart
        label="Memory"
        deviceTag="RAM"
        points={mem}
        stroke="#64748b"
        fill="#64748b"
        fillOpacity={0.1}
      />
    </div>
  );
}

export default HardwareCharts;
