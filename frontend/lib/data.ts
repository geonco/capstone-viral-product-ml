// Mock 데이터 로더 — public/mock 디렉토리의 JSON을 서버 컴포넌트에서 직접 읽음
// 실서비스 시 fetch('/api/...')로 갈아끼우기만 하면 됨
import { promises as fs } from "fs";
import path from "path";

const MOCK = path.join(process.cwd(), "public", "mock");

export type Summary = {
  keyword: string;
  category: string;
  growth_10d: number;
  sustainability_10d: number;
  buzz_composite_10d: number;
  crash_10d: number;
  spike_10d: number;
  fw_peak_softpos_10d: number;
  recent_avg: number;
  prev_avg: number;
  delta_pct: number;
};

export type SeriesPoint = { date: string; value: number | null };

export type Detail = {
  keyword: string;
  category: string;
  timeseries: {
    search: SeriesPoint[];
    click: SeriesPoint[];
    blog: SeriesPoint[];
    instagram: SeriesPoint[];
  };
  demographics: {
    gender: { male: number; female: number };
    ages: Record<string, number>;
  };
  predictions: Record<string, number>;
  shap: { feature: string; value: number }[];
  related: { keyword: string; lag_days: number; sync_rate: number }[];
};

export async function getSummary(): Promise<Summary[]> {
  const buf = await fs.readFile(path.join(MOCK, "summary.json"), "utf-8");
  return JSON.parse(buf);
}

export async function getDetail(keyword: string): Promise<Detail | null> {
  try {
    const buf = await fs.readFile(path.join(MOCK, "keywords", `${keyword}.json`), "utf-8");
    return JSON.parse(buf);
  } catch {
    return null;
  }
}

export async function listKeywords(): Promise<string[]> {
  const dir = path.join(MOCK, "keywords");
  const files = await fs.readdir(dir);
  return files.filter(f => f.endsWith(".json")).map(f => f.replace(/\.json$/, ""));
}
