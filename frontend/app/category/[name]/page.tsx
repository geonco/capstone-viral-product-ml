import { getSummary, getDetail } from "@/lib/data";
import { RankingTable } from "@/components/RankingTable";
import Link from "next/link";

const CATEGORIES = ["베이커리", "쿠키", "초콜릿/캔디", "한과/젤리", "스낵"];

export default async function CategoryPage({ params }: { params: { name: string } }) {
  const name = decodeURIComponent(params.name);
  const summary = await getSummary();
  const filtered = summary.filter(s => s.category === name).sort((a, b) => b.growth_10d - a.growth_10d);

  const sparkData: Record<string, { value: number | null }[]> = {};
  await Promise.all(filtered.map(async (s) => {
    const d = await getDetail(s.keyword);
    if (d) sparkData[s.keyword] = d.timeseries.search.slice(-60).map(p => ({ value: p.value }));
  }));

  const avgGrowth = filtered.length ? filtered.reduce((a, b) => a + b.growth_10d, 0) / filtered.length : 0;
  const avgSustain = filtered.length ? filtered.reduce((a, b) => a + b.sustainability_10d, 0) / filtered.length : 0;

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">카테고리 · {name}</h1>
        <p className="text-sub text-sm mt-1">{filtered.length}개 키워드 · 평균 성장 ×{avgGrowth.toFixed(2)} · 평균 지속성 {avgSustain.toFixed(2)}</p>
      </header>

      <nav className="flex gap-2 flex-wrap">
        {CATEGORIES.map(c => (
          <Link
            key={c}
            href={`/category/${encodeURIComponent(c)}`}
            className={`text-sm px-3 py-1.5 rounded-md border ${c === name ? "bg-accent/20 text-accent border-accent/40" : "border-border text-sub hover:bg-panel2"}`}
          >
            {c}
          </Link>
        ))}
      </nav>

      {filtered.length > 0 ? (
        <RankingTable rows={filtered} metric="growth_10d" sparkData={sparkData} />
      ) : (
        <div className="card p-8 text-center text-sub">이 카테고리에 키워드가 없음</div>
      )}
    </div>
  );
}
