"use client";
import { ResponsiveContainer, AreaChart, Area } from "recharts";

export function Sparkline({ data, color = "#7c5cff" }: { data: { value: number | null }[]; color?: string }) {
  const cleaned = data.map((d, i) => ({ i, value: d.value ?? 0 }));
  return (
    <ResponsiveContainer width="100%" height={36}>
      <AreaChart data={cleaned} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id={`spark-${color}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.5} />
            <stop offset="100%" stopColor={color} stopOpacity={0} />
          </linearGradient>
        </defs>
        <Area type="monotone" dataKey="value" stroke={color} strokeWidth={1.5} fill={`url(#spark-${color})`} />
      </AreaChart>
    </ResponsiveContainer>
  );
}
