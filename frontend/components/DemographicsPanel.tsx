"use client";
import { ResponsiveContainer, PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, Tooltip } from "recharts";

const AGE_LABELS: Record<string, string> = {
  age_10: "10대", age_20: "20대", age_30: "30대", age_40: "40대", age_50: "50대", age_60: "60대+",
};

export function DemographicsPanel({
  gender, ages,
}: {
  gender: { male: number; female: number };
  ages: Record<string, number>;
}) {
  const genderData = [
    { name: "남성", value: gender.male, color: "#22d3ee" },
    { name: "여성", value: gender.female, color: "#f472b6" },
  ];
  const ageData = Object.entries(ages).map(([k, v]) => ({ name: AGE_LABELS[k] ?? k, value: v }));

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="card p-4">
        <div className="text-xs text-sub mb-2">성별 클릭 비율 (최근 30일 평균)</div>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie data={genderData} dataKey="value" innerRadius={50} outerRadius={80} paddingAngle={2}>
              {genderData.map((d) => <Cell key={d.name} fill={d.color} />)}
            </Pie>
            <Tooltip contentStyle={{ background: "#11141b", border: "1px solid #222734", fontSize: 12 }} />
          </PieChart>
        </ResponsiveContainer>
        <div className="flex justify-around text-xs">
          {genderData.map((d) => (
            <div key={d.name} className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full" style={{ background: d.color }} />
              <span className="text-sub">{d.name}</span>
              <span className="font-medium">{d.value.toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card p-4">
        <div className="text-xs text-sub mb-2">연령대별 클릭 비율 (최근 30일 평균)</div>
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={ageData} margin={{ top: 8, right: 8, bottom: 0, left: -20 }}>
            <XAxis dataKey="name" stroke="#9aa3b2" fontSize={11} />
            <YAxis stroke="#9aa3b2" fontSize={11} />
            <Tooltip contentStyle={{ background: "#11141b", border: "1px solid #222734", fontSize: 12 }} />
            <Bar dataKey="value" fill="#7c5cff" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
