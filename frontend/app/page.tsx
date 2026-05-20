import { getSummary, getDetail, listKeywords, type Summary } from "@/lib/data";
import { SearchBar } from "@/components/SearchBar";
import { RankingTable } from "@/components/RankingTable";
import { Flame } from "lucide-react";

type ClusterFmt = "growth" | "sust" | "peak" | "crash" | "spike" | "magnitude" | "ratio";

function sortBy<T>(arr: T[], scoreFn: (x: T) => number, desc = true): T[] {
  return [...arr].sort((a, b) => {
    const sa = scoreFn(a);
    const sb = scoreFn(b);
    return desc ? sb - sa : sa - sb;
  });
}

function num(v: number | null | undefined, fallback = 0): number {
  return v == null || Number.isNaN(v) ? fallback : v;
}

export default async function HomePage() {
  const summary = await getSummary();
  const keywords = await listKeywords();

  const sparkData: Record<string, { value: number | null }[]> = {};
  await Promise.all(summary.map(async (s) => {
    const d = await getDetail(s.keyword);
    if (d) sparkData[s.keyword] = d.timeseries.search.slice(-60).map(p => ({ value: p.value }));
  }));

  // 8개 클러스터 — 각 라벨/조합 기준 상위 8개씩
  const trending  = sortBy(summary, x => num(x.growth_10d)).slice(0, 8);
  const sustain   = sortBy(summary, x => num(x.sustainability_10d)).slice(0, 8);
  const peakSoon  = sortBy(summary, x => num(x.fw_peak_softpos_10d, 99), false).slice(0, 8);
  const crashRisk = sortBy(summary, x => num(x.crash_10d)).slice(0, 8);

  // 고성장 + 저지속 — growth_10d 상위 ∩ sustainability_10d 하위. 점수 = growth × (2 - sust)
  const fastFlash = sortBy(summary, x =>
    num(x.growth_10d) * Math.max(0, 2 - num(x.sustainability_10d, 1))
  ).slice(0, 8);

  // 저변동 스테디 — sustainability 상위 ∩ fw_cv_10d 낮음. 점수 = sust / (cv + 0.1)
  const steady = sortBy(summary, x =>
    num(x.sustainability_10d) / (num(x.fw_cv_10d, 0.5) + 0.1)
  ).slice(0, 8);

  // 스파이크 유망 — spike_10d 상위 ∩ buzz_composite_10d 양수. 점수 = spike × max(buzz, 0)
  const sparkSurge = sortBy(summary, x =>
    num(x.spike_10d) * Math.max(0, num(x.buzz_composite_10d))
  ).slice(0, 8);

  // 매그니튜드 킹 — LSTM fw_magnitude_10d 절대 크기 상위
  const magKing = sortBy(summary, x => num(x.fw_magnitude_10d)).slice(0, 8);

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
        <RankingTable
          rows={sortBy(summary, x => num(x.growth_10d)).slice(0, 12)}
          metric="growth_10d"
          sparkData={sparkData}
        />
      </section>

      <section>
        <div className="flex items-baseline gap-2 mb-3">
          <h2 className="text-lg font-semibold">8가지 시그널 클러스터</h2>
          <span className="text-xs text-sub">라벨·조합별 상위 8개</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-5">
          <Cluster title="🔥 급상승 예상"     hint="growth_10d 상위"                   rows={trending}   fmt="growth"    />
          <Cluster title="💎 지속성 강자"     hint="sustainability_10d 상위"           rows={sustain}    fmt="sust"      />
          <Cluster title="⏰ 피크 임박"        hint="fw_peak_softpos_10d 작음 (LSTM)"   rows={peakSoon}   fmt="peak"      />
          <Cluster title="⚠️ 하락 주의"        hint="crash_10d 상위"                    rows={crashRisk}  fmt="crash"     />
          <Cluster title="🚀 고성장·저지속"    hint="단기 폭발형 (growth × (2 − sust))" rows={fastFlash}  fmt="ratio"     />
          <Cluster title="🛡️ 저변동 스테디"    hint="sust ÷ (fw_cv + 0.1)"              rows={steady}     fmt="sust"      />
          <Cluster title="⚡ 스파이크 유망"     hint="spike × max(buzz, 0)"              rows={sparkSurge} fmt="spike"     />
          <Cluster title="👑 매그니튜드 킹"     hint="fw_magnitude_10d 상위 (LSTM)"      rows={magKing}    fmt="magnitude" />
        </div>
      </section>
    </div>
  );
}

function formatScore(r: Summary, fmt: ClusterFmt): string {
  switch (fmt) {
    case "growth":    return `×${num(r.growth_10d).toFixed(2)}`;
    case "sust":      return `${num(r.sustainability_10d).toFixed(2)}`;
    case "peak":      return `${num(r.fw_peak_softpos_10d, 99).toFixed(2)}`;
    case "crash":     return `${(num(r.crash_10d) * 100).toFixed(0)}%`;
    case "spike":     return `${num(r.spike_10d).toFixed(2)}`;
    case "magnitude": return `${num(r.fw_magnitude_10d).toFixed(2)}`;
    case "ratio":     return `×${num(r.growth_10d).toFixed(1)}`;
  }
}

function Cluster({ title, hint, rows, fmt }: { title: string; hint: string; rows: Summary[]; fmt: ClusterFmt }) {
  return (
    <div className="card p-5">
      <div className="flex items-baseline gap-2 mb-3">
        <h3 className="font-semibold text-sm">{title}</h3>
      </div>
      <div className="text-[11px] text-sub mb-2">{hint}</div>
      <ul className="divide-y divide-border">
        {rows.map((r, i) => (
          <li key={r.keyword} className="flex items-center py-1.5 gap-2">
            <span className="text-sub text-xs w-4 tabular-nums">{i + 1}</span>
            <a href={`/keyword/${encodeURIComponent(r.keyword)}`} className="flex-1 hover:text-accent text-sm truncate">{r.keyword}</a>
            <span className="text-[10px] tabular-nums text-sub min-w-[44px] text-right">{formatScore(r, fmt)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
