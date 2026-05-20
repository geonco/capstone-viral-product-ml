import { getDetail, getSummary } from "@/lib/data";
import { notFound } from "next/navigation";
import Link from "next/link";
import { TimeSeriesChart } from "@/components/TimeSeriesChart";
import { DemographicsPanel } from "@/components/DemographicsPanel";
import { RadarPanel } from "@/components/RadarPanel";
import { WindowSwitcher } from "@/components/WindowSwitcher";
import { ForecastChart } from "@/components/ForecastChart";
import { StagedDonut } from "@/components/StagedDonut";
import { LstmGauges } from "@/components/LstmGauges";
import { RiskOpportunityBar } from "@/components/RiskOpportunityBar";
import { KeywordSummary } from "@/components/KeywordSummary";

export default async function KeywordPage({ params }: { params: { name: string } }) {
  const keyword = decodeURIComponent(params.name);
  const detail = await getDetail(keyword);
  const summary = await getSummary();
  if (!detail) notFound();

  const sum = summary.find(s => s.keyword === keyword);
  const matrix = detail.predictions_lgbm?.matrix ?? {};
  const lstmPreds = detail.predictions_lstm ?? {};
  const growthProbs = detail.predictions_lgbm?.staged_probs?.growth ?? {};
  const buzzProbs   = detail.predictions_lgbm?.staged_probs?.buzz_composite ?? {};

  const peakSoftpos = {
    "10": lstmPreds.fw_peak_softpos_10d ?? null,
    "15": lstmPreds.fw_peak_softpos_15d ?? null,
  };

  // 전체 키워드의 crash/spike 분포 — 분위 계산용
  const allCrash = summary.map(s => s.crash_10d).filter((x): x is number => x != null);
  const allSpike = summary.map(s => s.spike_10d).filter((x): x is number => x != null);

  return (
    <div className="space-y-8">
      <header className="flex flex-wrap items-end gap-3">
        <Link href="/" className="text-sub text-sm hover:text-text">← 랭킹</Link>
        <h1 className="text-3xl font-bold tracking-tight">{keyword}</h1>
        <span className="chip">{detail.category}</span>
        {detail.data_reference_date && (
          <span className="text-xs text-sub px-2 py-0.5 rounded border border-border">
            기준일 <span className="font-mono">{detail.data_reference_date}</span>
          </span>
        )}
        <span className="text-sm text-sub ml-auto">
          최근 30일 평균: {sum?.recent_avg.toLocaleString() ?? "—"} 검색
          {sum?.delta_pct != null && (
            <span className={`ml-2 ${sum.delta_pct > 0 ? "text-good" : sum.delta_pct < 0 ? "text-bad" : ""}`}>
              ({sum.delta_pct > 0 ? "+" : ""}{sum.delta_pct.toFixed(1)}%)
            </span>
          )}
        </span>
      </header>

      <KeywordSummary
        keyword={keyword}
        category={detail.category}
        matrix={matrix}
        growthProbs={growthProbs}
        buzzProbs={buzzProbs}
        peakSoftpos15={lstmPreds.fw_peak_softpos_15d ?? null}
      />

      <Section title="윈도우별 핵심 지표" hint="5d / 10d / 15d 토글">
        <div className="card p-5">
          <WindowSwitcher matrix={matrix} peakSoftpos={peakSoftpos} />
        </div>
      </Section>

      <Section title="시계열 추이 + 미래 예측" hint="실측 60일 + seq2seq 미래 15일">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <div className="card p-5">
            <div className="text-xs text-sub mb-2">4채널 실측</div>
            <TimeSeriesChart
              series={[
                { name: "검색", color: "#7c5cff", data: detail.timeseries.search },
                { name: "쇼핑클릭", color: "#22d3ee", data: detail.timeseries.click },
                { name: "블로그", color: "#34d399", data: detail.timeseries.blog },
                { name: "인스타", color: "#f472b6", data: detail.timeseries.instagram },
              ]}
            />
          </div>
          <div className="card p-5">
            <div className="text-xs text-sub mb-2">합산 신호 + seq2seq 예측</div>
            <ForecastChart
              history={mergeChannels(detail.timeseries)}
              forecast={detail.forecast_series ?? []}
            />
          </div>
        </div>
      </Section>

      <Section title="staged 분류 확률" hint="LightGBM 분류기 출력 — 어느 구간에 속할 가능성이 큰가">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="card p-5">
            <StagedDonut title="growth_10d" probs={growthProbs.growth_10d ?? {}} order={["shrink", "stable", "surge", "extreme"]} />
          </div>
          <div className="card p-5">
            <StagedDonut title="buzz_composite_10d" probs={buzzProbs.buzz_composite_10d ?? {}} order={["negative", "neutral", "positive"]} />
          </div>
          <div className="card p-5">
            <div className="text-xs text-sub mb-3">위험 vs 기회 — 전체 키워드 분위</div>
            <RiskOpportunityBar
              crash={matrix.crash?.["10"] ?? null}
              spike={matrix.spike?.["10"] ?? null}
              allCrash={allCrash}
              allSpike={allSpike}
            />
          </div>
        </div>
      </Section>

      <Section title="패턴 형상 + 라벨 매트릭스" hint="레이더 + 윈도우 × 라벨 매트릭스 + LSTM 게이지">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="card p-5">
            <div className="text-xs text-sub mb-2">패턴 형상 (10d 정규화)</div>
            <RadarPanel predictions={detail.predictions} />
          </div>
          <div className="card p-5 lg:col-span-2 overflow-x-auto">
            <div className="text-xs text-sub mb-2">윈도우별 예측 매트릭스</div>
            <PredictionGrid matrix={matrix} />
          </div>
        </div>
        <div className="card p-5 mt-4">
          <div className="text-xs text-sub mb-3">LSTM 보조 라벨 — magnitude / log_growth / peak_softpos / spike / cv / decline / delta</div>
          <LstmGauges preds={lstmPreds} />
        </div>
      </Section>

      <Section title="인구통계" hint="누가 클릭하는가 — 최근 30일">
        <DemographicsPanel gender={detail.demographics.gender} ages={detail.demographics.ages} />
      </Section>
    </div>
  );
}

function Section({ title, hint, children }: { title: string; hint: string; children: React.ReactNode }) {
  return (
    <section>
      <div className="flex items-baseline gap-2 mb-3">
        <h2 className="text-lg font-semibold">{title}</h2>
        <span className="text-xs text-sub">{hint}</span>
      </div>
      {children}
    </section>
  );
}

function PredictionGrid({ matrix }: { matrix: Record<string, Record<string, number | null>> }) {
  const labels = [
    { key: "growth",         name: "성장 (×배수)" },
    { key: "sustainability", name: "지속성" },
    { key: "buzz_composite", name: "버즈 z-score" },
    { key: "spike",          name: "급등" },
    { key: "crash",          name: "하락 (%)" },
  ];
  const wins = ["5", "10", "15"];
  return (
    <table className="w-full text-sm">
      <thead className="text-sub text-xs uppercase">
        <tr>
          <th className="text-left py-2">라벨</th>
          {wins.map(w => <th key={w} className="text-right py-2 px-3">{w}d</th>)}
        </tr>
      </thead>
      <tbody>
        {labels.map(l => (
          <tr key={l.key} className="border-t border-border">
            <td className="py-2.5">{l.name}</td>
            {wins.map(w => {
              const v = matrix[l.key]?.[w];
              const display = v == null
                ? "—"
                : l.key === "crash"
                  ? `${(v * 100).toFixed(0)}%`
                  : v.toFixed(2);
              return <td key={w} className="text-right tabular-nums px-3">{display}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function mergeChannels(ts: { search: { date: string; value: number | null }[]; click: { date: string; value: number | null }[]; blog: { date: string; value: number | null }[]; instagram: { date: string; value: number | null }[] }) {
  // 4채널 합산 시계열 — seq2seq forecast가 합산 신호 스케일이라 비교 가능
  const map = new Map<string, number>();
  for (const ch of [ts.search, ts.click, ts.blog, ts.instagram]) {
    for (const p of ch) {
      if (p.value == null) continue;
      map.set(p.date, (map.get(p.date) ?? 0) + p.value);
    }
  }
  return Array.from(map.entries())
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([date, value]) => ({ date, value }));
}
