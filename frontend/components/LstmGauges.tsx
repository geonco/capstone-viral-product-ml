// LSTM 11~15개 라벨을 한 줄 게이지로 — 절대값 + 정규화 막대
const LABELS: { key: string; name: string; max: number; min?: number; fmt?: (v: number) => string; tone?: "good" | "bad" | "neutral" }[] = [
  { key: "fw_magnitude_5d",    name: "magnitude 5d",  max: 10 },
  { key: "fw_magnitude_10d",   name: "magnitude 10d", max: 10 },
  { key: "fw_magnitude_15d",   name: "magnitude 15d", max: 10 },
  { key: "fw_log_growth_10d",  name: "log_growth 10d", min: -2, max: 3, tone: "good" },
  { key: "fw_log_growth_15d",  name: "log_growth 15d", min: -2, max: 3, tone: "good" },
  { key: "fw_peak_softpos_10d", name: "peak_softpos 10d", max: 1 },
  { key: "fw_peak_softpos_15d", name: "peak_softpos 15d", max: 1 },
  { key: "fw_spike_10d",       name: "spike 10d",     max: 3 },
  { key: "fw_spike_15d",       name: "spike 15d",     max: 3 },
  { key: "fw_cv_10d",          name: "cv 10d",        max: 1 },
  { key: "fw_cv_15d",          name: "cv 15d",        max: 1 },
  { key: "fw_decline_10d",     name: "decline 10d",   max: 1, tone: "bad" },
  { key: "fw_decline_15d",     name: "decline 15d",   max: 1, tone: "bad" },
  { key: "fw_delta_10d",       name: "delta 10d",     min: -5, max: 10 },
  { key: "fw_delta_15d",       name: "delta 15d",     min: -5, max: 10 },
];

export function LstmGauges({ preds }: { preds: Record<string, number> }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-1.5">
      {LABELS.map(l => {
        const v = preds[l.key];
        if (v == null) return null;
        const min = l.min ?? 0;
        const norm = Math.max(0, Math.min(1, (v - min) / (l.max - min)));
        const color = l.tone === "bad" ? "#f87171" : l.tone === "good" ? "#34d399" : "#7c5cff";
        return (
          <div key={l.key} className="flex items-center gap-3 text-xs">
            <span className="w-32 text-sub truncate">{l.name}</span>
            <div className="flex-1 h-1.5 rounded-full bg-panel2 overflow-hidden">
              <div className="h-full rounded-full" style={{ width: `${norm * 100}%`, background: color }} />
            </div>
            <span className="w-14 text-right tabular-nums">{v.toFixed(2)}</span>
          </div>
        );
      })}
    </div>
  );
}
