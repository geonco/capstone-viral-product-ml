"use client";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";

export function ShapBar({ data }: { data: { feature: string; value: number }[] }) {
  return (
    <ResponsiveContainer width="100%" height={Math.max(260, data.length * 22)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 12, bottom: 4, left: 130 }}>
        <XAxis type="number" stroke="#9aa3b2" fontSize={11} />
        <YAxis type="category" dataKey="feature" stroke="#9aa3b2" fontSize={11} width={130} />
        <Tooltip contentStyle={{ background: "#11141b", border: "1px solid #222734", fontSize: 12 }} />
        <Bar dataKey="value" fill="#22d3ee" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
