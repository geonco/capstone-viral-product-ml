// crash / spike의 키워드 분위 — 전체 분포에서 어느 위치인지 표시
export function RiskOpportunityBar({
  crash, spike, allCrash, allSpike,
}: {
  crash: number | null;
  spike: number | null;
  allCrash: number[];
  allSpike: number[];
}) {
  function quantile(arr: number[], v: number | null): number | null {
    if (v == null || arr.length === 0) return null;
    const sorted = [...arr].filter(x => x != null && !Number.isNaN(x)).sort((a, b) => a - b);
    const idx = sorted.findIndex(x => x >= v);
    return idx < 0 ? 1 : idx / sorted.length;
  }
  const qCrash = quantile(allCrash, crash);
  const qSpike = quantile(allSpike, spike);

  function Row({ label, value, q, color }: { label: string; value: number | null; q: number | null; color: string }) {
    return (
      <div className="space-y-1">
        <div className="flex items-baseline justify-between">
          <span className="text-xs text-sub">{label}</span>
          <span className="text-xs tabular-nums">
            {value == null ? "—" : value.toFixed(3)}
            {q != null && <span className="text-sub ml-2">상위 {((1 - q) * 100).toFixed(0)}%</span>}
          </span>
        </div>
        <div className="relative h-2 rounded-full bg-panel2 overflow-hidden">
          <div className="absolute inset-y-0 left-0" style={{ width: q != null ? `${q * 100}%` : "0%", background: color }} />
          {q != null && (
            <div className="absolute top-0 bottom-0 w-px bg-white/60" style={{ left: `${q * 100}%` }} />
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <Row label="하락 위험 (crash_10d)"  value={crash} q={qCrash} color="#f87171" />
      <Row label="급등 기회 (spike_10d)"  value={spike} q={qSpike} color="#34d399" />
      <div className="text-[11px] text-sub">전체 키워드 499개 분포 기준 분위 — 색칠된 영역이 이 키워드의 위치</div>
    </div>
  );
}
