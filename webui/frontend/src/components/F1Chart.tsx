import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { LANDCOVER_CLASSES, LANDCOVER_COLORS } from '@/constants';
import type { Metrics } from '@/types';

interface F1ChartProps {
  metrics: Metrics;
}

// Build color lookup: class name -> hex color
const CLASS_COLOR_MAP: Record<string, string> = Object.fromEntries(
  Object.entries(LANDCOVER_CLASSES).map(([idx, name]) => [
    name,
    LANDCOVER_COLORS[Number(idx)] ?? '#888888',
  ]),
);

export function F1Chart({ metrics }: F1ChartProps) {
  const perClass = metrics?.per_class;
  if (!perClass || typeof perClass !== 'object') return null;

  const data = Object.entries(perClass).map(([className, scores]) => ({
    name: className,
    f1: Math.round(scores.f1 * 1000) / 10, // e.g., 0.9147 -> 91.5
    fill: CLASS_COLOR_MAP[className] ?? '#888888',
  }));

  if (data.length === 0) return null;

  return (
    <div>
      <h4 className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">
        Per-Class F1 Score
      </h4>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data}>
          <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} angle={-20} textAnchor="end" />
          <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 10 }} />
          <Tooltip formatter={(val: number) => `${val.toFixed(1)}%`} />
          <Bar dataKey="f1" radius={[2, 2, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={index} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
