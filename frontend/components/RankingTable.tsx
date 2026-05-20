import Link from "next/link";
import { Summary } from "@/lib/data";
import { Sparkline } from "./Sparkline";

export function RankingTable({
  rows,
  metric,
  sparkData,
}: {
  rows: Summary[];
  metric: keyof Summary;
  sparkData?: Record<string, { value: number | null }[]>;
}) {
  return (
    <div className="card overflow-hidden">
      <table className="w-full text-sm">
        <thead className="bg-panel2 text-sub text-xs uppercase tracking-wide">
          <tr>
            <th className="text-left px-4 py-3 w-10">#</th>
            <th className="text-left px-4 py-3">키워드</th>
            <th className="text-left px-4 py-3 hidden md:table-cell">카테고리</th>
            <th className="text-left px-4 py-3 hidden lg:table-cell">최근 추세</th>
            <th className="text-right px-4 py-3">성장</th>
            <th className="text-right px-4 py-3 hidden sm:table-cell">지속성</th>
            <th className="text-right px-4 py-3 hidden md:table-cell">피크(d)</th>
            <th className="text-right px-4 py-3">변화</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.keyword} className="border-t border-border hover:bg-panel2/40">
              <td className="px-4 py-3 text-sub">{i + 1}</td>
              <td className="px-4 py-3">
                <Link href={`/keyword/${encodeURIComponent(r.keyword)}`} className="font-medium hover:text-accent">
                  {r.keyword}
                </Link>
              </td>
              <td className="px-4 py-3 hidden md:table-cell">
                <span className="chip">{r.category}</span>
              </td>
              <td className="px-4 py-3 hidden lg:table-cell w-32">
                {sparkData?.[r.keyword] && (
                  <Sparkline data={sparkData[r.keyword]} color={r.delta_pct >= 0 ? "#34d399" : "#f87171"} />
                )}
              </td>
              <td className="px-4 py-3 text-right tabular-nums">×{(r.growth_10d ?? 0).toFixed(2)}</td>
              <td className="px-4 py-3 text-right tabular-nums hidden sm:table-cell">{(r.sustainability_10d ?? 0).toFixed(2)}</td>
              <td className="px-4 py-3 text-right tabular-nums hidden md:table-cell">{(r.fw_peak_softpos_10d ?? 0).toFixed(2)}</td>
              <td className={`px-4 py-3 text-right tabular-nums ${r.delta_pct >= 0 ? "text-good" : "text-bad"}`}>
                {r.delta_pct >= 0 ? "+" : ""}{r.delta_pct.toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
