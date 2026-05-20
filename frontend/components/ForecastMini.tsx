// 카드형 작은 forecast 차트 — small multiples 용
type Point = { date: string; value: number | null };

export function ForecastMini({ forecast }: { forecast: Point[] }) {
  if (!forecast.length) return <div className="h-12 flex items-center text-xs text-sub">예측 없음</div>;
  const vals = forecast.map(p => p.value ?? 0);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = Math.max(max - min, 1);
  const w = 120, h = 36;
  const pts = vals.map((v, i) => {
    const x = (i / (vals.length - 1)) * w;
    const y = h - ((v - min) / span) * h;
    return `${x},${y}`;
  }).join(" ");
  const last = vals[vals.length - 1];
  const first = vals[0];
  const delta = last - first;
  const color = delta > 0 ? "#34d399" : delta < 0 ? "#f87171" : "#9ca3af";
  return (
    <div className="flex items-center gap-2">
      <svg width={w} height={h} className="overflow-visible">
        <polyline fill="none" stroke={color} strokeWidth={1.5} points={pts} />
      </svg>
      <span className="text-[10px] tabular-nums text-sub whitespace-nowrap">
        {first.toFixed(0)}→{last.toFixed(0)}
      </span>
    </div>
  );
}
