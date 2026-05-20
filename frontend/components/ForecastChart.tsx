"use client";
import { useMemo } from "react";
import {
  ResponsiveContainer, ComposedChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend, ReferenceLine,
} from "recharts";

type Point = { date: string; value: number | null };

export function ForecastChart({ history, forecast, lookbackDays = 60 }: {
  history: Point[];
  forecast: Point[];
  lookbackDays?: number;
}) {
  const merged = useMemo(() => {
    const hist = history.slice(-lookbackDays);
    const cutoff = hist.length ? hist[hist.length - 1].date : "";
    const rows: { date: string; hist: number | null; fcst: number | null }[] = [];
    for (const p of hist) {
      rows.push({ date: p.date, hist: p.value, fcst: null });
    }
    // 마지막 historical 값을 미래 첫 점에 연결해 끊김 방지
    const lastHist = hist.length ? hist[hist.length - 1].value : null;
    let first = true;
    for (const p of forecast) {
      rows.push({
        date: p.date,
        hist: null,
        fcst: first ? lastHist ?? p.value : p.value,
      });
      first = false;
    }
    return { rows, cutoff };
  }, [history, forecast, lookbackDays]);

  return (
    <div>
      <div className="flex items-center justify-between mb-2 text-xs text-sub">
        <span>과거 {lookbackDays}일 (실선) + seq2seq 예측 {forecast.length}일 (점선)</span>
        <span className="text-[11px]">합산 신호 (search+click+blog+insta) 스케일</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <ComposedChart data={merged.rows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="#222734" strokeDasharray="3 3" />
          <XAxis dataKey="date" stroke="#9aa3b2" fontSize={11} tickFormatter={(d) => d.slice(5)} minTickGap={32} />
          <YAxis stroke="#9aa3b2" fontSize={11} />
          <Tooltip
            contentStyle={{ background: "#11141b", border: "1px solid #222734", borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: "#9aa3b2" }}
            formatter={(v: number) => v?.toFixed(1) ?? "—"}
          />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          {merged.cutoff && (
            <ReferenceLine x={merged.cutoff} stroke="#7c5cff" strokeDasharray="2 4" strokeOpacity={0.6}
              label={{ value: "now", fill: "#7c5cff", fontSize: 10, position: "top" }} />
          )}
          <Line name="실측" type="monotone" dataKey="hist" stroke="#22d3ee" strokeWidth={1.8} dot={false} connectNulls={false} />
          <Line name="예측" type="monotone" dataKey="fcst" stroke="#f472b6" strokeWidth={1.8} strokeDasharray="5 4" dot={false} connectNulls={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
