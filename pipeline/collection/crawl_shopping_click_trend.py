# 네이버 데이터랩 쇼핑인사이트 오픈API로 키워드별 일별 쇼핑 클릭 트렌드 수집
# 선정된 키워드 CSV를 읽어 키워드×일 매트릭스 CSV 생성
# 키워드별 개별 조회 (교차 정규화 방지), API 에러 시 즉시 중단

import os
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# .env 로딩 (최소 의존성: python-dotenv 미사용)
_env = PROJECT_ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

with open(PROJECT_ROOT / "configs" / "collection.yaml", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)["click_trend"]

API_URL    = _cfg["api_url"]
CATEGORY   = _cfg["category"]
START_DATE = _cfg["start_date"]
END_DATE   = _cfg["end_date"]
TIME_UNIT  = _cfg["time_unit"]
SLEEP_SEC  = _cfg["sleep_seconds"]
INPUT_CSV  = _cfg["input_csv"]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "shopping_click"

CLIENT_ID     = os.environ["NAVER_CLIENT_ID"]
CLIENT_SECRET = os.environ["NAVER_CLIENT_SECRET"]

HEADERS = {
    "X-Naver-Client-Id": CLIENT_ID,
    "X-Naver-Client-Secret": CLIENT_SECRET,
    "Content-Type": "application/json",
}

# 단일 키워드의 전체 기간 쇼핑 클릭 트렌드 조회
def fetch_keyword_trend(keyword):
    body = {
        "startDate": START_DATE,
        "endDate": END_DATE,
        "timeUnit": TIME_UNIT,
        "category": CATEGORY,
        "keyword": [{"name": keyword, "param": [keyword]}],
    }

    resp = requests.post(API_URL, headers=HEADERS, json=body)
    resp.raise_for_status()

    results = resp.json().get("results", [])
    if not results or not results[0].get("data"):
        return {}
    return {d["period"]: d["ratio"] for d in results[0]["data"]}


def main():
    # 키워드 로딩
    input_path = PROJECT_ROOT / "data" / "raw" / "keyword_select" / INPUT_CSV
    keywords = pd.read_csv(input_path, encoding="utf-8-sig")["keyword"].tolist()

    # 기존 결과 로딩 (이어서 수집)
    output_path = OUTPUT_DIR / "trend.csv"
    if output_path.exists():
        existing = pd.read_csv(output_path, index_col="keyword", encoding="utf-8-sig")
        done = set(existing.index)
    else:
        existing = pd.DataFrame()
        done = set()

    todo = [kw for kw in keywords if kw not in done]
    if not todo:
        return
    print(f"shopping_click_trend: {len(todo)}/{len(keywords)} keywords to collect")

    # 키워드별 쇼핑 클릭 트렌드 수집 (중단 시 중간 결과 저장)
    rows = {}
    try:
        for i, kw in enumerate(todo, 1):
            trend = fetch_keyword_trend(kw)
            rows[kw] = trend
            print(f"[{i}/{len(todo)}] {kw}: {len(trend)}d")
            time.sleep(SLEEP_SEC)
    finally:
        if rows:
            df_new = pd.DataFrame.from_dict(rows, orient="index")
            df = pd.concat([existing, df_new])
            df.index.name = "keyword"
            df = df.reindex(sorted(df.columns), axis=1)
            df.to_csv(output_path, encoding="utf-8-sig")
            print(f"saved: {output_path.name} ({df.shape[0]}x{df.shape[1]})")


if __name__ == "__main__":
    main()
