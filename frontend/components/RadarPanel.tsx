"use client";
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from "recharts";

export function RadarPanel({ predictions }: { predictions: Record<string, number> }) {
  // 5종 라벨을 0~1 정규화하여 레이더에 매핑 — 절대 비교가 아닌 모양 파악용
  const norm = (v: number, hi: number) => Math.max(0, Math.min(1, v / hi));
  const data = [
    { axis: "성장", value: norm((predictions.growth_10d ?? 1) - 1 + 0.5, 1.5) },
    { axis: "지속성", value: norm(predictions.sustainability_10d ?? 0, 2) },
    { axis: "버즈", value: norm((predictions.buzz_composite_10d ?? 0) + 5, 10) },
    { axis: "급등", value: norm(predictions.spike_10d ?? 0, 1) },
    { axis: "하락 위험", value: norm(predictions.crash_10d ?? 0, 1) },
  ];
  return (
    <ResponsiveContainer width="100%" height={260}>
      <RadarChart data={data}>
        <PolarGrid stroke="#222734" />
        <PolarAngleAxis dataKey="axis" stroke="#9aa3b2" fontSize={11} />
        <PolarRadiusAxis stroke="#222734" tick={false} />
        <Radar dataKey="value" stroke="#7c5cff" fill="#7c5cff" fillOpacity={0.35} />
      </RadarChart>
    </ResponsiveContainer>
  );
}
