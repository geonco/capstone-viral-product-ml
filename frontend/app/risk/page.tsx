import { getSummary } from "@/lib/data";
import Link from "next/link";

export default async function RiskPage() {
  const summary = await getSummary();

  // 점수 — crash_15d(높을수록 위험) ÷ sustainability_15d(낮을수록 위험)
  const scored = summary
    .filter(s => s.crash_15d != null && s.sustainability_15d != null)
    .map(s => ({
      ...s,
      risk_score: ((s.crash_15d ?? 0) * 100) / Math.max(0.1, s.sustainability_15d ?? 1),
    }))
    .sort((a, b) => b.risk_score - a.risk_score);

  const top = scored.slice(0, 30);
  const safe = [...scored].sort((a, b) => a.risk_score - b.risk_score).slice(0, 12);

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">하락 위험 워치리스트</h1>
        <p className="text-sub text-sm mt-2">
          15일 윈도우 crash 예측 ÷ sustainability — 큰 값일수록 단기간 큰 폭락 위험
        </p>
      </header>

      <section>
        <div className="text-xs text-sub mb-3">위험 상위 30</div>
        <div className="card overflow-hidden">
          <table className="w-full text-sm">
            <thead className="bg-panel2 text-sub text-xs uppercase">
              <tr>
                <th className="text-left px-4 py-2 w-10">#</th>
                <th className="text-left px-4 py-2">키워드</th>
                <th className="text-left px-4 py-2">카테고리</th>
                <th className="text-right px-4 py-2">crash_15d</th>
                <th className="text-right px-4 py-2">sust_15d</th>
                <th className="text-right px-4 py-2">risk score</th>
                <th className="text-right px-4 py-2">최근 30일 평균</th>
              </tr>
            </thead>
            <tbody>
              {top.map((r, i) => (
                <tr key={r.keyword} className="border-t border-border hover:bg-panel2/40">
                  <td className="px-4 py-2 text-sub text-xs tabular-nums">{i + 1}</td>
                  <td className="px-4 py-2">
                    <Link href={`/keyword/${encodeURIComponent(r.keyword)}`} className="hover:text-accent">{r.keyword}</Link>
                  </td>
                  <td className="px-4 py-2"><span className="chip">{r.category}</span></td>
                  <td className="px-4 py-2 text-right tabular-nums text-bad">{((r.crash_15d ?? 0) * 100).toFixed(0)}%</td>
                  <td className="px-4 py-2 text-right tabular-nums">{(r.sustainability_15d ?? 0).toFixed(2)}</td>
                  <td className="px-4 py-2 text-right tabular-nums font-semibold">{r.risk_score.toFixed(1)}</td>
                  <td className="px-4 py-2 text-right tabular-nums text-sub">{r.recent_avg.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <div className="text-xs text-sub mb-3">반대편 — 가장 안전한 키워드 12개 (참고용)</div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
          {safe.map(s => (
            <Link key={s.keyword} href={`/keyword/${encodeURIComponent(s.keyword)}`}
              className="card p-3 hover:border-good/40 flex items-center justify-between">
              <span className="text-sm truncate">{s.keyword}</span>
              <span className="text-[10px] text-good tabular-nums">{s.risk_score.toFixed(1)}</span>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
