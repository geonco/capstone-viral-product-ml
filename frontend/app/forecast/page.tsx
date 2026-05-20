import { getSummary, getDetail } from "@/lib/data";
import Link from "next/link";
import { ForecastMini } from "@/components/ForecastMini";

export default async function ForecastPage() {
  const summary = await getSummary();

  // 키워드별 forecast_series 미리 로드 — fw_delta_10d (LSTM) 큰 순으로 정렬
  const enriched = await Promise.all(summary.map(async (s) => {
    const d = await getDetail(s.keyword);
    const forecast = d?.forecast_series ?? [];
    const first = forecast[0]?.value ?? 0;
    const last  = forecast[forecast.length - 1]?.value ?? 0;
    return {
      keyword: s.keyword,
      category: s.category,
      growth: s.growth_10d ?? 0,
      forecast,
      first,
      last,
      delta_abs: last - first,
      delta_pct: first > 0 ? ((last - first) / first) * 100 : 0,
    };
  }));

  const ranked = enriched
    .filter(e => e.forecast.length > 0)
    .sort((a, b) => b.delta_pct - a.delta_pct);

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">미래 15일 궤적</h1>
        <p className="text-sub text-sm mt-2">
          seq2seq 모델이 예측한 키워드별 합산 신호(검색+클릭+블로그+인스타) 미래 15일 — 시작값 대비 변화율 큰 순
        </p>
      </header>

      <section>
        <div className="text-xs text-sub mb-3">상승 예상 상위 — 시작 → 끝 변화율 기준</div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
          {ranked.slice(0, 60).map((e, i) => (
            <Link key={e.keyword} href={`/keyword/${encodeURIComponent(e.keyword)}`}
              className="card p-3 hover:border-accent/40 transition-colors">
              <div className="flex items-center justify-between mb-1">
                <span className="text-sub text-xs tabular-nums w-5">{i + 1}</span>
                <span className="text-sm font-medium flex-1 ml-2 truncate">{e.keyword}</span>
                <span className="chip text-[10px]">{e.category}</span>
              </div>
              <ForecastMini forecast={e.forecast} />
              <div className="mt-1 text-[11px] text-sub tabular-nums">
                Δ {e.delta_pct > 0 ? "+" : ""}{e.delta_pct.toFixed(1)}%
              </div>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
