// Mock 데이터 로더 — public/mock 디렉토리의 JSON을 서버 컴포넌트에서 직접 읽음
// 실서비스 시 fetch('/api/...')로 갈아끼우기만 하면 됨
import { promises as fs } from "fs";
import path from "path";

const MOCK = path.join(process.cwd(), "public", "mock");

export type Summary = {
  keyword: string;
  category: string;
  growth_10d: number | null;
  sustainability_10d: number | null;
  buzz_composite_10d: number | null;
  crash_10d: number | null;
  spike_10d: number | null;
  growth_5d?: number | null;
  growth_15d?: number | null;
  sustainability_5d?: number | null;
  sustainability_15d?: number | null;
  crash_15d?: number | null;
  fw_peak_softpos_10d: number | null;
  fw_peak_softpos_15d?: number | null;
  fw_magnitude_10d?: number | null;
  fw_cv_10d?: number | null;
  fw_log_growth_10d?: number | null;
  fw_decline_10d?: number | null;
  recent_avg: number;
  prev_avg: number;
  delta_pct: number;
};

export type SeriesPoint = { date: string; value: number | null };

export type Detail = {
  keyword: string;
  category: string;
  data_reference_date?: string;
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
  predictions_lgbm?: {
    matrix: Record<string, Record<string, number | null>>;
    staged_probs: {
      growth?: Record<string, Record<string, number>>;
      buzz_composite?: Record<string, Record<string, number>>;
    };
  };
  predictions_lstm?: Record<string, number>;
  forecast_series?: SeriesPoint[];
  shap: { feature: string; value: number }[];
  related: { keyword: string; lag_days: number; sync_rate: number }[];
};

type SummaryFile = Summary[] | { data_reference_date: string; keywords: Summary[] };

let _refDate: string | null = null;

export async function getSummary(): Promise<Summary[]> {
  const buf = await fs.readFile(path.join(MOCK, "summary.json"), "utf-8");
  const parsed = JSON.parse(buf) as SummaryFile;
  if (Array.isArray(parsed)) {
    _refDate = null;
    return parsed;
  }
  _refDate = parsed.data_reference_date;
  return parsed.keywords;
}

export async function getDataReferenceDate(): Promise<string | null> {
  if (_refDate !== null) return _refDate;
  try {
    const buf = await fs.readFile(path.join(MOCK, "summary.json"), "utf-8");
    const parsed = JSON.parse(buf) as SummaryFile;
    if (!Array.isArray(parsed)) {
      _refDate = parsed.data_reference_date;
      return _refDate;
    }
  } catch {
    /* noop */
  }
  return null;
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
