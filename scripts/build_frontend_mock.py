# 프론트엔드 mock 데이터 생성 — raw csv에서 20개 키워드 추출 후 JSON 저장
# 목적: Next.js public/mock과 Streamlit data 디렉토리에 동일 JSON을 떨궈 두 프론트가 같은 데이터로 비교되게 함

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT_FRONT = ROOT / "frontend" / "public" / "mock"
OUT_STREAM = ROOT / "streamlit_app" / "mock"

OUT_FRONT.mkdir(parents=True, exist_ok=True)
OUT_STREAM.mkdir(parents=True, exist_ok=True)

# 시각화용 최근 윈도우 — 365일치 일별 시계열만 노출
DAYS = 365
TOP_N = 20

random.seed(42)
np.random.seed(42)


def load_wide(path):
    # wide csv 로드 — keyword 행 × 날짜 열, utf-8-sig BOM 처리
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={df.columns[0]: "keyword"})
    return df


def tail_series(df, keyword, days):
    # 키워드 한 개의 마지막 days일 시계열을 (date, value) 리스트로 변환
    row = df.loc[df["keyword"] == keyword]
    if row.empty:
        return []
    s = row.iloc[0, 1:].astype(float)
    s = s.tail(days)
    return [
        {"date": str(d), "value": (None if (pd.isna(v)) else float(v))}
        for d, v in s.items()
    ]


def safe_mean(arr):
    a = [x for x in arr if x is not None and not math.isnan(x)]
    return float(np.mean(a)) if a else 0.0


def pick_top_keywords(search_df, n):
    # 최근 60일 검색량 합 상위 n개 — 데모에 즐길거리가 많은 키워드 우선
    last_cols = list(search_df.columns[-60:])
    score = search_df[last_cols].sum(axis=1)
    idx = score.sort_values(ascending=False).head(n).index
    return search_df.loc[idx, "keyword"].tolist()


def fake_predictions(search_series, click_series):
    # 라벨 mock — 실제 모델 추론 대신 최근 추세에서 그럴듯한 값 합성
    s_vals = [p["value"] for p in search_series if p["value"] is not None]
    if len(s_vals) < 30:
        return None
    recent = np.array(s_vals[-14:])
    prev = np.array(s_vals[-30:-14])
    growth_ratio = float((recent.mean() + 1) / (prev.mean() + 1))
    # 라벨 5종(buzz/growth/sustain/crash/spike) × 윈도우 3개
    out = {}
    for w in [5, 10, 15]:
        scale = 1 + (w - 10) * 0.05
        out[f"growth_{w}d"] = round(growth_ratio * scale, 3)
        out[f"sustainability_{w}d"] = round(min(3.0, growth_ratio * 0.9 * scale), 3)
        out[f"buzz_composite_{w}d"] = round(np.tanh((growth_ratio - 1) * 2) * 5, 3)
        out[f"crash_{w}d"] = round(max(0.0, 1 - growth_ratio) * 0.6, 3)
        out[f"spike_{w}d"] = round(min(1.0, max(0.0, growth_ratio - 1)), 3)
    # LSTM 측 라벨
    out["fw_peak_softpos_10d"] = round(random.uniform(2, 9), 1)
    out["fw_peak_softpos_15d"] = round(random.uniform(3, 12), 1)
    out["fw_cv_10d"] = round(np.std(recent) / (np.mean(recent) + 1), 3)
    return out


def fake_shap(features_meta):
    # SHAP top 15 mock — 라벨별 모델 학습이 끝나기 전 데모용 합성
    candidates = [
        ("search_growth_14d", 0.8),
        ("click_growth_14d", 0.7),
        ("mention_growth_7d", 0.6),
        ("search_pos_30d", 0.55),
        ("viral_force", 0.5),
        ("engagement_force", 0.45),
        ("click_search_ratio", 0.4),
        ("search_slope_7d", 0.38),
        ("mention_search_ratio", 0.35),
        ("buzz_composite_lookback", 0.32),
        ("age20_click_ratio", 0.28),
        ("female_click_ratio", 0.25),
        ("month_sin", 0.18),
        ("days_since_max_14d", 0.15),
        ("click_lead_signal_7d", 0.12),
    ]
    return [
        {"feature": name, "value": round(v * (0.7 + 0.6 * random.random()), 3)}
        for name, v in candidates
    ]


def build_related(all_keywords, current):
    # 연관 키워드 mock — cluster_leadlag 결과 대신 임의 5개 + 시차/동조율
    pool = [k for k in all_keywords if k != current]
    chosen = random.sample(pool, k=min(8, len(pool)))
    out = []
    for k in chosen:
        out.append({
            "keyword": k,
            "lag_days": random.randint(-5, 5),
            "sync_rate": round(random.uniform(0.55, 0.92), 3),
        })
    return sorted(out, key=lambda x: -x["sync_rate"])


def main():
    print("loading raw csv ...")
    search = load_wide(RAW / "search" / "absolute.csv")
    click = load_wide(RAW / "shopping_click" / "absolute.csv")
    male = load_wide(RAW / "shopping_click" / "gender_m_pct.csv")
    female = load_wide(RAW / "shopping_click" / "gender_f_pct.csv")
    age_dfs = {
        "age_10": load_wide(RAW / "shopping_click" / "age_10_pct.csv"),
        "age_20": load_wide(RAW / "shopping_click" / "age_20_pct.csv"),
        "age_30": load_wide(RAW / "shopping_click" / "age_30_pct.csv"),
        "age_40": load_wide(RAW / "shopping_click" / "age_40_pct.csv"),
        "age_50": load_wide(RAW / "shopping_click" / "age_50_pct.csv"),
        "age_60": load_wide(RAW / "shopping_click" / "age_60_pct.csv"),
    }

    sometrend = pd.read_csv(RAW / "sometrends" / "sometrend_mention_long.csv", encoding="utf-8-sig")
    sometrend["date"] = pd.to_datetime(sometrend["date"]).astype(str)

    keywords = pick_top_keywords(search, TOP_N)
    print(f"selected: {keywords}")

    # 카테고리 mock — 실제 카테고리 메타가 없으니 키워드 패턴 기반 임의 분류
    def categorize(k):
        if any(t in k for t in ["빵", "식빵", "베이글", "크로와상", "카스테라"]):
            return "베이커리"
        if any(t in k for t in ["쿠키", "비스킷", "마카롱"]):
            return "쿠키"
        if any(t in k for t in ["초콜릿", "초코", "캬라멜", "캔디", "사탕"]):
            return "초콜릿/캔디"
        if any(t in k for t in ["젤리", "약과", "한과", "오란다", "강정"]):
            return "한과/젤리"
        return "스낵"

    summary = []
    for kw in keywords:
        s_series = tail_series(search, kw, DAYS)
        c_series = tail_series(click, kw, DAYS)
        m_row = male.loc[male["keyword"] == kw]
        f_row = female.loc[female["keyword"] == kw]
        gender = {
            "male": safe_mean(m_row.iloc[0, 1:].astype(float).tail(30).tolist()) if not m_row.empty else 50.0,
            "female": safe_mean(f_row.iloc[0, 1:].astype(float).tail(30).tolist()) if not f_row.empty else 50.0,
        }
        ages = {}
        for label, df in age_dfs.items():
            row = df.loc[df["keyword"] == kw]
            ages[label] = safe_mean(row.iloc[0, 1:].astype(float).tail(30).tolist()) if not row.empty else 0.0

        st_kw = sometrend[sometrend["keyword"] == kw].sort_values("date").tail(DAYS)
        blog_series = [
            {"date": d, "value": float(v) if not pd.isna(v) else None}
            for d, v in zip(st_kw["date"], st_kw.get("블로그", pd.Series(dtype=float)))
        ]
        insta_series = [
            {"date": d, "value": float(v) if not pd.isna(v) else None}
            for d, v in zip(st_kw["date"], st_kw.get("인스타그램", pd.Series(dtype=float)))
        ]

        preds = fake_predictions(s_series, c_series) or {}
        detail = {
            "keyword": kw,
            "category": categorize(kw),
            "timeseries": {
                "search": s_series,
                "click": c_series,
                "blog": blog_series,
                "instagram": insta_series,
            },
            "demographics": {
                "gender": gender,
                "ages": ages,
            },
            "predictions": preds,
            "shap": fake_shap(None),
            "related": build_related(keywords, kw),
        }

        path_front = OUT_FRONT / "keywords"
        path_front.mkdir(exist_ok=True)
        path_stream = OUT_STREAM / "keywords"
        path_stream.mkdir(exist_ok=True)

        body = json.dumps(detail, ensure_ascii=False)
        (path_front / f"{kw}.json").write_text(body, encoding="utf-8")
        (path_stream / f"{kw}.json").write_text(body, encoding="utf-8")

        s_vals = [p["value"] for p in s_series if p["value"] is not None]
        last30 = s_vals[-30:] if len(s_vals) >= 30 else s_vals
        prev30 = s_vals[-60:-30] if len(s_vals) >= 60 else s_vals
        recent_avg = float(np.mean(last30)) if last30 else 0.0
        prev_avg = float(np.mean(prev30)) if prev30 else 0.0

        summary.append({
            "keyword": kw,
            "category": detail["category"],
            "growth_10d": preds.get("growth_10d", 1.0),
            "sustainability_10d": preds.get("sustainability_10d", 1.0),
            "buzz_composite_10d": preds.get("buzz_composite_10d", 0.0),
            "crash_10d": preds.get("crash_10d", 0.0),
            "spike_10d": preds.get("spike_10d", 0.0),
            "fw_peak_softpos_10d": preds.get("fw_peak_softpos_10d", 5.0),
            "recent_avg": round(recent_avg, 1),
            "prev_avg": round(prev_avg, 1),
            "delta_pct": round((recent_avg / max(1.0, prev_avg) - 1) * 100, 1),
        })

    body = json.dumps(summary, ensure_ascii=False)
    (OUT_FRONT / "summary.json").write_text(body, encoding="utf-8")
    (OUT_STREAM / "summary.json").write_text(body, encoding="utf-8")
    print(f"wrote {len(summary)} keywords + summary.json")


if __name__ == "__main__":
    main()
