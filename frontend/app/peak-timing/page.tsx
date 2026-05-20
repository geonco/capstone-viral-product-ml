import { getSummary } from "@/lib/data";
import Link from "next/link";

export default async function PeakTimingPage() {
  const summary = await getSummary();

  // fw_peak_softpos_10d ∈ [0, 1] — 10일 윈도우 안에서 피크의 상대 위치
  // 일수 환산: ceil(softpos * 10) 일
  const enriched = summary
    .filter(s => s.fw_peak_softpos_10d != null)
    .map(s => ({
      ...s,
      peak_day: Math.max(1, Math.round((s.fw_peak_softpos_10d ?? 0) * 10)),
      peak_day_15: s.fw_peak_softpos_15d != null
        ? Math.max(1, Math.round((s.fw_peak_softpos_15d ?? 0) * 15))
        : null,
    }))
    .sort((a, b) => (a.fw_peak_softpos_10d ?? 1) - (b.fw_peak_softpos_10d ?? 1));

  const within3 = enriched.filter(e => e.peak_day <= 3);
  const within7 = enriched.filter(e => e.peak_day > 3 && e.peak_day <= 7);
  const within14 = enriched.filter(e => e.peak_day > 7);

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">피크 시점 예측</h1>
        <p className="text-sub text-sm mt-2">
          LSTM의 fw_peak_softpos — 향후 10일 안에 검색량 피크가 언제 올지 예측 (0=오늘, 1=10일 후)
        </p>
      </header>

      <Bucket title="🔥 3일 안 피크 예상" hint="가장 임박한 키워드" rows={within3} accent="bad" />
      <Bucket title="⏳ 1주 안 피크 예상" hint="4~7일" rows={within7} accent="warn" />
      <Bucket title="📅 2주 안 피크 예상" hint="8~14일" rows={within14} accent="neutral" />
    </div>
  );
}

function Bucket({ title, hint, rows, accent }: { title: string; hint: string; rows: any[]; accent: string }) {
  return (
    <section>
      <div className="flex items-baseline gap-2 mb-3">
        <h2 className="text-lg font-semibold">{title}</h2>
        <span className="text-xs text-sub">{hint}</span>
        <span className="text-xs text-sub ml-auto">{rows.length}개</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
        {rows.slice(0, 24).map(r => (
          <Link key={r.keyword} href={`/keyword/${encodeURIComponent(r.keyword)}`}
            className="card p-3 hover:border-accent/40 flex items-center gap-3">
            <span className={`tabular-nums text-${accent} font-semibold w-8 text-center`}>{r.peak_day}d</span>
            <span className="flex-1 text-sm truncate">{r.keyword}</span>
            <span className="chip text-[10px]">{r.category}</span>
            <span className="text-[10px] text-sub tabular-nums">
              g{(r.growth_10d ?? 1).toFixed(1)}
            </span>
          </Link>
        ))}
        {rows.length > 24 && (
          <div className="card p-3 text-center text-xs text-sub">+ {rows.length - 24}개 더</div>
        )}
      </div>
    </section>
  );
}
