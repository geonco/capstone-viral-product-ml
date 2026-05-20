"use client";

type Probs = Record<string, number>;

const COLORS: Record<string, string> = {
  shrink: "#f87171", stable: "#fbbf24", surge: "#34d399", extreme: "#22d3ee",
  negative: "#f87171", neutral: "#9ca3af", positive: "#34d399",
};

function arc(cx: number, cy: number, r: number, start: number, end: number) {
  // start/end in radians
  const x1 = cx + r * Math.cos(start);
  const y1 = cy + r * Math.sin(start);
  const x2 = cx + r * Math.cos(end);
  const y2 = cy + r * Math.sin(end);
  const large = end - start > Math.PI ? 1 : 0;
  return `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} Z`;
}

export function StagedDonut({ title, probs, order }: {
  title: string;
  probs: Probs;
  order?: string[];
}) {
  const keys = order ?? Object.keys(probs);
  const total = keys.reduce((s, k) => s + (probs[k] || 0), 0) || 1;
  const size = 140, cx = size / 2, cy = size / 2, r = 56, ir = 32;
  let acc = -Math.PI / 2;
  const slices = keys.map(k => {
    const frac = (probs[k] || 0) / total;
    const start = acc;
    const end = acc + frac * Math.PI * 2;
    acc = end;
    return { k, frac, start, end };
  });
  const dominant = keys.reduce((b, k) => (probs[k] || 0) > (probs[b] || 0) ? k : b, keys[0]);

  return (
    <div className="flex items-center gap-4">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {slices.map(s => (
          s.frac > 0.001 && (
            <path key={s.k} d={arc(cx, cy, r, s.start, s.end)} fill={COLORS[s.k] || "#7c5cff"} opacity={0.85} />
          )
        ))}
        <circle cx={cx} cy={cy} r={ir} fill="#11141b" />
        <text x={cx} y={cy - 4} textAnchor="middle" fontSize="9" fill="#9aa3b2">최다</text>
        <text x={cx} y={cy + 9} textAnchor="middle" fontSize="11" fontWeight="600" fill={COLORS[dominant]}>{dominant}</text>
      </svg>
      <div className="space-y-1 text-xs flex-1">
        <div className="text-sub mb-1">{title}</div>
        {keys.map(k => (
          <div key={k} className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-sm" style={{ background: COLORS[k] || "#7c5cff" }} />
            <span className="w-16">{k}</span>
            <span className="tabular-nums text-sub">{((probs[k] || 0) * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
