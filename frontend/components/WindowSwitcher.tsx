"use client";
import { useState } from "react";
import { KpiCard } from "@/components/KpiCard";

type Matrix = Record<string, Record<string, number | null>>;

export function WindowSwitcher({ matrix, peakSoftpos }: {
  matrix: Matrix;
  peakSoftpos: { "10": number | null; "15": number | null };
}) {
  const [w, setW] = useState<"5" | "10" | "15">("10");

  const g  = matrix.growth?.[w];
  const s  = matrix.sustainability?.[w];
  const b  = matrix.buzz_composite?.[w];
  const sp = matrix.spike?.[w];
  const cr = matrix.crash?.[w];
  const peak = w === "5" ? null : peakSoftpos[w as "10" | "15"];

  const growthTone = g == null ? "neutral" : g > 1.1 ? "good" : g < 0.95 ? "bad" : "neutral";
  const crashTone  = cr == null ? "neutral" : cr > 0.4 ? "bad" : cr > 0.2 ? "warn" : "neutral";

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <span className="text-xs text-sub">윈도우</span>
        {(["5", "10", "15"] as const).map(k => (
          <button
            key={k}
            onClick={() => setW(k)}
            className={`text-xs px-2.5 py-1 rounded-md ${
              w === k ? "bg-accent/20 text-accent" : "text-sub hover:bg-panel2"
            }`}
          >
            {k}d
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <KpiCard label={`성장 ×배수 (${w}d)`}      value={g == null ? "—" : g.toFixed(2)} delta={g == null ? undefined : (g - 1) * 100} tone={growthTone} hint={`growth_${w}d`} />
        <KpiCard label={`지속성 (${w}d)`}           value={s == null ? "—" : s.toFixed(2)} unit="x"             hint={`sustainability_${w}d`} />
        <KpiCard label={`버즈 (${w}d)`}             value={b == null ? "—" : b.toFixed(2)} hint={`buzz_composite_${w}d`} />
        <KpiCard label={`급등 (${w}d)`}             value={sp == null ? "—" : sp.toFixed(2)} hint={`spike_${w}d`} />
        <KpiCard label={`하락 (${w}d)`}             value={cr == null ? "—" : `${(cr * 100).toFixed(0)}%`} tone={crashTone} hint={`crash_${w}d`} />
      </div>
      {peak != null && (
        <div className="text-xs text-sub pl-1">
          피크 시점(LSTM): {peak.toFixed(2)} (0~1 정규화 — 0에 가까울수록 윈도우 앞쪽, 1에 가까울수록 뒤쪽)
        </div>
      )}
    </div>
  );
}
