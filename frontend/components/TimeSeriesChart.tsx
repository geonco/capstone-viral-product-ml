"use client";
import { useMemo, useState } from "react";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend,
} from "recharts";

type Series = { name: string; color: string; data: { date: string; value: number | null }[] };

export function TimeSeriesChart({ series, days = 180 }: { series: Series[]; days?: number }) {
  const [window, setWindow] = useState<number>(days);
  const [scale, setScale] = useState<"linear" | "log">("linear");

  const merged = useMemo(() => {
    const map = new Map<string, any>();
    for (const s of series) {
      const sliced = s.data.slice(-window);
      for (const p of sliced) {
        const row = map.get(p.date) ?? { date: p.date };
        row[s.name] = p.value;
        map.set(p.date, row);
      }
    }
    return Array.from(map.values()).sort((a, b) => a.date.localeCompare(b.date));
  }, [series, window]);

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <div className="flex gap-1">
          {[30, 90, 180, 365].map((d) => (
            <button
              key={d}
              onClick={() => setWindow(d)}
              className={`text-xs px-2.5 py-1 rounded-md ${
                window === d ? "bg-accent/20 text-accent" : "text-sub hover:bg-panel2"
              }`}
            >
              {d}d
            </button>
          ))}
        </div>
        <div className="flex gap-1">
          {(["linear", "log"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setScale(s)}
              className={`text-xs px-2.5 py-1 rounded-md ${
                scale === s ? "bg-accent/20 text-accent" : "text-sub hover:bg-panel2"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={merged} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#222734" strokeDasharray="3 3" />
          <XAxis dataKey="date" stroke="#9aa3b2" fontSize={11} tickFormatter={(d) => d.slice(5)} minTickGap={32} />
          <YAxis stroke="#9aa3b2" fontSize={11} scale={scale} domain={scale === "log" ? [1, "auto"] : ["auto", "auto"]} />
          <Tooltip
            contentStyle={{ background: "#11141b", border: "1px solid #222734", borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: "#9aa3b2" }}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {series.map((s) => (
            <Line key={s.name} type="monotone" dataKey={s.name} stroke={s.color} strokeWidth={1.6} dot={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
