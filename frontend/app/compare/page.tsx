import { getSummary, getDetail, listKeywords } from "@/lib/data";
import { TimeSeriesChart } from "@/components/TimeSeriesChart";
import Link from "next/link";

export default async function ComparePage({ searchParams }: { searchParams: { keywords?: string } }) {
  const all = await listKeywords();
  const summary = await getSummary();
  const selected = (searchParams.keywords?.split(",").filter(Boolean) ?? all.slice(0, 3));
  const details = await Promise.all(selected.map(k => getDetail(k)));
  const valid = details.filter((d): d is NonNullable<typeof d> => !!d);

  const palette = ["#7c5cff", "#22d3ee", "#34d399", "#f472b6", "#fbbf24"];
  const series = valid.map((d, i) => ({
    name: d.keyword, color: palette[i % palette.length], data: d.timeseries.search,
  }));

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold tracking-tight">키워드 비교</h1>
        <p className="text-sub text-sm mt-1">최대 5개 키워드 시계열·KPI 동시 비교</p>
      </header>

      <div className="card p-5">
        <div className="text-xs text-sub mb-3">현재 선택: {valid.map(v => v.keyword).join(" · ") || "(없음)"}</div>
        <div className="flex flex-wrap gap-2">
          {all.map(k => {
            const isSel = selected.includes(k);
            const next = isSel ? selected.filter(s => s !== k) : [...selected, k].slice(0, 5);
            return (
              <Link
                key={k}
                href={`/compare?keywords=${encodeURIComponent(next.join(","))}`}
                className={`text-xs px-2.5 py-1.5 rounded-md border ${isSel ? "bg-accent/20 text-accent border-accent/40" : "border-border text-sub hover:bg-panel2"}`}
              >
                {k}
              </Link>
            );
          })}
        </div>
      </div>

      {valid.length > 0 && (
        <>
          <div className="card p-5">
            <div className="text-xs text-sub mb-2">검색량 시계열 비교</div>
            <TimeSeriesChart series={series} />
          </div>

          <div className="card overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-panel2 text-sub text-xs uppercase">
                <tr>
                  <th className="text-left px-4 py-3">키워드</th>
                  <th className="text-right px-4 py-3">성장</th>
                  <th className="text-right px-4 py-3">지속성</th>
                  <th className="text-right px-4 py-3">버즈</th>
                  <th className="text-right px-4 py-3">피크(d)</th>
                  <th className="text-right px-4 py-3">하락</th>
                </tr>
              </thead>
              <tbody>
                {valid.map((d) => {
                  const s = summary.find(x => x.keyword === d.keyword)!;
                  return (
                    <tr key={d.keyword} className="border-t border-border">
                      <td className="px-4 py-3 font-medium">{d.keyword}</td>
                      <td className="px-4 py-3 text-right tabular-nums">×{(s.growth_10d ?? 0).toFixed(2)}</td>
                      <td className="px-4 py-3 text-right tabular-nums">{(s.sustainability_10d ?? 0).toFixed(2)}</td>
                      <td className="px-4 py-3 text-right tabular-nums">{(s.buzz_composite_10d ?? 0).toFixed(2)}</td>
                      <td className="px-4 py-3 text-right tabular-nums">{(s.fw_peak_softpos_10d ?? 0).toFixed(2)}</td>
                      <td className="px-4 py-3 text-right tabular-nums">{((s.crash_10d ?? 0) * 100).toFixed(0)}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
