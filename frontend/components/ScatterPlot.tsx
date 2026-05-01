"use client";
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceLine, Cell } from "recharts";
import { useRouter } from "next/navigation";

export function ScatterPlot({ data }: { data: any[] }) {
  const router = useRouter();
  const points = data.map(d => ({ ...d, x: d.growth_10d, y: d.sustainability_10d }));
  return (
    <ResponsiveContainer width="100%" height={420}>
      <ScatterChart margin={{ top: 16, right: 16, bottom: 30, left: 0 }}>
        <CartesianGrid stroke="#222734" strokeDasharray="3 3" />
        <XAxis type="number" dataKey="x" stroke="#9aa3b2" fontSize={11} domain={["auto", "auto"]} label={{ value: "성장 (growth_10d)", position: "bottom", fill: "#9aa3b2", fontSize: 11 }} />
        <YAxis type="number" dataKey="y" stroke="#9aa3b2" fontSize={11} label={{ value: "지속성", angle: -90, position: "insideLeft", fill: "#9aa3b2", fontSize: 11 }} />
        <ReferenceLine x={1.05} stroke="#444a5a" strokeDasharray="4 4" />
        <ReferenceLine y={1.0} stroke="#444a5a" strokeDasharray="4 4" />
        <Tooltip
          cursor={{ strokeDasharray: "3 3" }}
          contentStyle={{ background: "#11141b", border: "1px solid #222734", fontSize: 12 }}
          formatter={(v: any, name: string) => [Number(v).toFixed(2), name]}
          labelFormatter={(_, p: any) => p?.[0]?.payload?.keyword ?? ""}
        />
        <Scatter
          data={points}
          onClick={(e: any) => router.push(`/keyword/${encodeURIComponent(e.keyword)}`)}
        >
          {points.map((p, i) => (
            <Cell key={i} fill={p.x >= 1.05 && p.y >= 1.0 ? "#34d399" : p.x >= 1.05 ? "#fbbf24" : p.y >= 1.0 ? "#22d3ee" : "#f87171"} />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
}
