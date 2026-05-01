import { getDetail, getSummary } from "@/lib/data";
import { notFound } from "next/navigation";
import Link from "next/link";
import { KpiCard } from "@/components/KpiCard";
import { TimeSeriesChart } from "@/components/TimeSeriesChart";
import { DemographicsPanel } from "@/components/DemographicsPanel";
import { ShapBar } from "@/components/ShapBar";
import { RadarPanel } from "@/components/RadarPanel";

export default async function KeywordPage({ params }: { params: { name: string } }) {
  const keyword = decodeURIComponent(params.name);
  const detail = await getDetail(keyword);
  const summary = await getSummary();
  if (!detail) notFound();

  const sum = summary.find(s => s.keyword === keyword);
  const p = detail.predictions;
  const peak = p.fw_peak_softpos_10d ?? 0;
  const peakTone = peak <= 4 ? "good" : peak <= 8 ? "warn" : "neutral";
  const growthTone = (p.growth_10d ?? 1) > 1.1 ? "good" : (p.growth_10d ?? 1) < 0.95 ? "bad" : "neutral";

  return (
    <div className="space-y-8">
      <header className="flex flex-wrap items-end gap-3">
        <Link href="/" className="text-sub text-sm hover:text-text">← 랭킹</Link>
        <h1 className="text-3xl font-bold tracking-tight">{keyword}</h1>
        <span className="chip">{detail.category}</span>
        <span className="text-sm text-sub ml-auto">최근 30일 평균: {sum?.recent_avg.toLocaleString()} 검색</span>
      </header>

      <section className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard label="14일 성장 전망 (×배수)" value={(p.growth_10d ?? 1).toFixed(2)} delta={((p.growth_10d ?? 1) - 1) * 100} tone={growthTone} hint="growth_10d 앙상블" />
        <KpiCard label="지속성" value={(p.sustainability_10d ?? 1).toFixed(2)} unit="x" hint="sustainability_10d" />
        <KpiCard label="피크 시점" value={peak.toFixed(1)} unit="일 후" tone={peakTone} hint="fw_peak_softpos_10d" />
        <KpiCard label="현재 버즈" value={(p.buzz_composite_10d ?? 0).toFixed(2)} hint="buzz_composite z-score" />
      </section>

      <Tabs detail={detail} />
    </div>
  );
}

function Tabs({ detail }: { detail: NonNullable<Awaited<ReturnType<typeof getDetail>>> }) {
  return (
    <div className="space-y-8">
      <Section title="시계열 추이" hint="검색 / 클릭 / 블로그 / 인스타 4채널">
        <div className="card p-5">
          <TimeSeriesChart
            series={[
              { name: "검색", color: "#7c5cff", data: detail.timeseries.search },
              { name: "쇼핑클릭", color: "#22d3ee", data: detail.timeseries.click },
              { name: "블로그", color: "#34d399", data: detail.timeseries.blog },
              { name: "인스타", color: "#f472b6", data: detail.timeseries.instagram },
            ]}
          />
        </div>
      </Section>

      <Section title="종합 패턴" hint="5종 라벨 레이더 + 예측 표">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="card p-5">
            <div className="text-xs text-sub mb-2">패턴 형상 (10d 윈도우 정규화)</div>
            <RadarPanel predictions={detail.predictions} />
          </div>
          <div className="card p-5 lg:col-span-2 overflow-x-auto">
            <div className="text-xs text-sub mb-2">윈도우별 예측값 — 5d / 10d / 15d</div>
            <PredictionGrid predictions={detail.predictions} />
          </div>
        </div>
      </Section>

      <Section title="인구통계" hint="누가 클릭하는가 — 최근 30일">
        <DemographicsPanel gender={detail.demographics.gender} ages={detail.demographics.ages} />
      </Section>

      <Section title="모델 기여도 (SHAP top 15)" hint="LightGBM 예측에 영향을 준 피처">
        <div className="card p-5">
          <ShapBar data={detail.shap} />
        </div>
      </Section>

      <Section title="연관 키워드" hint="cluster_leadlag 기반 — 시차와 동조율">
        <div className="card overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-panel2 text-sub text-xs uppercase">
              <tr>
                <th className="text-left px-4 py-3">키워드</th>
                <th className="text-right px-4 py-3">시차 (일)</th>
                <th className="text-right px-4 py-3">동조율</th>
                <th className="text-right px-4 py-3">관계</th>
              </tr>
            </thead>
            <tbody>
              {detail.related.map((r) => (
                <tr key={r.keyword} className="border-t border-border hover:bg-panel2/40">
                  <td className="px-4 py-3">
                    <Link href={`/keyword/${encodeURIComponent(r.keyword)}`} className="hover:text-accent">{r.keyword}</Link>
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">{r.lag_days > 0 ? `+${r.lag_days}` : r.lag_days}</td>
                  <td className="px-4 py-3 text-right tabular-nums">{(r.sync_rate * 100).toFixed(0)}%</td>
                  <td className="px-4 py-3 text-right text-xs text-sub">
                    {r.lag_days < 0 ? "선행" : r.lag_days > 0 ? "후행" : "동시"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}

function Section({ title, hint, children }: any) {
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

function PredictionGrid({ predictions }: { predictions: Record<string, number> }) {
  const labels = [
    { key: "growth", name: "성장 (×배수)" },
    { key: "sustainability", name: "지속성" },
    { key: "buzz_composite", name: "버즈 z-score" },
    { key: "spike", name: "급등" },
    { key: "crash", name: "하락" },
  ];
  const wins = [5, 10, 15];
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
              const v = predictions[`${l.key}_${w}d`];
              return <td key={w} className="text-right tabular-nums px-3">{v?.toFixed(2) ?? "—"}</td>;
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
