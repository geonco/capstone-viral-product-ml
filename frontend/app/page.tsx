import { getSummary, getDetail, listKeywords } from "@/lib/data";
import { SearchBar } from "@/components/SearchBar";
import { RankingTable } from "@/components/RankingTable";
import { Flame, Gem, Clock, AlertTriangle } from "lucide-react";

export default async function HomePage() {
  const summary = await getSummary();
  const keywords = await listKeywords();

  // 스파크라인용 — 키워드별 검색 시계열 last 60일만 미리 로드
  const sparkData: Record<string, { value: number | null }[]> = {};
  await Promise.all(summary.map(async (s) => {
    const d = await getDetail(s.keyword);
    if (d) sparkData[s.keyword] = d.timeseries.search.slice(-60).map(p => ({ value: p.value }));
  }));

  const trending = [...summary].sort((a, b) => b.growth_10d - a.growth_10d).slice(0, 8);
  const sustain = [...summary].sort((a, b) => b.sustainability_10d - a.sustainability_10d).slice(0, 8);
  const peakSoon = [...summary].sort((a, b) => a.fw_peak_softpos_10d - b.fw_peak_softpos_10d).slice(0, 8);
  const crashRisk = [...summary].sort((a, b) => b.crash_10d - a.crash_10d).slice(0, 8);

  return (
    <div className="space-y-10">
      <section className="text-center pt-8 pb-4">
        <h1 className="text-4xl md:text-5xl font-bold tracking-tight">
          오늘 어떤 <span className="text-transparent bg-clip-text bg-gradient-to-br from-accent to-accent2">식품 키워드</span>가 뜰까
        </h1>
        <p className="mt-3 text-sub text-sm md:text-base">
          네이버 검색·쇼핑 클릭 + SomeTrend 언급량 + 인구통계를 결합한 14일 바이럴 예측
        </p>
        <div className="mt-7 max-w-2xl mx-auto">
          <SearchBar keywords={keywords} />
        </div>
      </section>

      <section>
        <div className="flex items-center gap-2 mb-3">
          <Flame size={18} className="text-bad" />
          <h2 className="text-lg font-semibold">오늘의 랭킹</h2>
          <span className="text-xs text-sub ml-2">growth_10d 상위 — LightGBM × LSTM 앙상블</span>
        </div>
        <RankingTable rows={summary.sort((a, b) => b.growth_10d - a.growth_10d).slice(0, 12)} metric="growth_10d" sparkData={sparkData} />
      </section>

      <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Cluster title="🔥 급상승 예상" hint="growth_10d" rows={trending} />
        <Cluster title="💎 지속성 강자" hint="sustainability_10d" rows={sustain} />
        <Cluster title="⏰ 피크 임박" hint="fw_peak_softpos_10d ≤ 5d" rows={peakSoon} reverseFmt />
        <Cluster title="⚠️ 하락 주의" hint="crash_10d" rows={crashRisk} crash />
      </section>
    </div>
  );
}

function Cluster({ title, hint, rows, reverseFmt, crash }: { title: string; hint: string; rows: any[]; reverseFmt?: boolean; crash?: boolean }) {
  return (
    <div className="card p-5">
      <div className="flex items-baseline gap-2 mb-3">
        <h3 className="font-semibold">{title}</h3>
        <span className="text-xs text-sub">{hint}</span>
      </div>
      <ul className="divide-y divide-border">
        {rows.map((r, i) => (
          <li key={r.keyword} className="flex items-center py-2 gap-3">
            <span className="text-sub text-xs w-4 tabular-nums">{i + 1}</span>
            <a href={`/keyword/${encodeURIComponent(r.keyword)}`} className="flex-1 hover:text-accent text-sm">{r.keyword}</a>
            <span className="chip">{r.category}</span>
            <span className="text-xs tabular-nums w-16 text-right text-sub">
              {reverseFmt
                ? `${r.fw_peak_softpos_10d.toFixed(1)}d`
                : crash
                  ? `${(r.crash_10d * 100).toFixed(0)}%`
                  : `×${r.growth_10d.toFixed(2)}`}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
