import { getSummary } from "@/lib/data";
import { ScatterPlot } from "@/components/ScatterPlot";

export default async function ExplorePage() {
  const summary = await getSummary();
  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">탐색</h1>
        <p className="text-sub text-sm mt-1">성장 × 지속성 사분면 — 스타급 / 롱런 / 반짝 / 하락</p>
      </header>
      <div className="card p-5">
        <ScatterPlot data={summary} />
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Quad title="🌟 스타급" desc="성장 ↑ 지속성 ↑" rows={summary.filter(s => (s.growth_10d ?? 0) >= 1.05 && (s.sustainability_10d ?? 0) >= 1.0)} />
        <Quad title="🐢 롱런" desc="성장 ≈ 지속성 ↑" rows={summary.filter(s => (s.growth_10d ?? 0) < 1.05 && (s.sustainability_10d ?? 0) >= 1.0)} />
        <Quad title="✨ 반짝" desc="성장 ↑ 지속성 ↓" rows={summary.filter(s => (s.growth_10d ?? 0) >= 1.05 && (s.sustainability_10d ?? 0) < 1.0)} />
        <Quad title="📉 하락" desc="성장 ↓ 지속성 ↓" rows={summary.filter(s => (s.growth_10d ?? 0) < 1.05 && (s.sustainability_10d ?? 0) < 1.0)} />
      </div>
    </div>
  );
}

function Quad({ title, desc, rows }: { title: string; desc: string; rows: any[] }) {
  return (
    <div className="card p-4">
      <div className="font-semibold text-sm">{title}</div>
      <div className="text-xs text-sub mb-2">{desc}</div>
      <ul className="space-y-1">
        {rows.slice(0, 6).map(r => (
          <li key={r.keyword}>
            <a href={`/keyword/${encodeURIComponent(r.keyword)}`} className="text-sm hover:text-accent">{r.keyword}</a>
          </li>
        ))}
        {rows.length === 0 && <li className="text-xs text-sub">—</li>}
      </ul>
    </div>
  );
}
