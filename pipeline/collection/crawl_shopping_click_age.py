# 네이버 데이터랩 쇼핑인사이트 오픈API로 키워드별 일별 연령별 쇼핑 클릭 트렌드 수집
# 키워드당 1회 호출로 전체 연령대 분해 결과를 받아 age_10.csv ~ age_60.csv 생성
# 키워드별 개별 조회, API 에러 시 즉시 중단

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
    _cfg = yaml.safe_load(f)["click_age"]

API_URL    = _cfg["api_url"]
CATEGORY   = _cfg["category"]
START_DATE = _cfg["start_date"]
END_DATE   = _cfg["end_date"]
TIME_UNIT  = _cfg["time_unit"]
SLEEP_SEC  = _cfg["sleep_seconds"]
INPUT_CSV  = _cfg["input_csv"]
GROUPS     = _cfg["groups"]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "shopping_click"

CLIENT_ID     = os.environ["NAVER_CLIENT_ID"]
CLIENT_SECRET = os.environ["NAVER_CLIENT_SECRET"]

HEADERS = {
    "X-Naver-Client-Id": CLIENT_ID,
    "X-Naver-Client-Secret": CLIENT_SECRET,
    "Content-Type": "application/json",
}


# 단일 키워드의 전체 기간 연령별 쇼핑 클릭 트렌드 조회
# 반환: {group_code: {period: ratio, ...}, ...}
def fetch_keyword_age(keyword):
    body = {
        "startDate": START_DATE,
        "endDate": END_DATE,
        "timeUnit": TIME_UNIT,
        "category": CATEGORY,
        "keyword": keyword,
        "device": "",
        "gender": "",
        "ages": [],
    }

    resp = requests.post(API_URL, headers=HEADERS, json=body)
    resp.raise_for_status()

    results = resp.json().get("results", [])
    if not results or not results[0].get("data"):
        return {}

    grouped = {}
    for d in results[0]["data"]:
        g = d["group"]
        grouped.setdefault(g, {})[d["period"]] = d["ratio"]
    return grouped


def main():
    # 키워드 로딩
    input_path = PROJECT_ROOT / "data" / "raw" / "keyword_select" / INPUT_CSV
    keywords = pd.read_csv(input_path, encoding="utf-8-sig")["keyword"].tolist()

    # 그룹별 기존 결과 로딩
    existing = {}
    done_sets = []
    for g in GROUPS:
        label = g["label"]
        path = OUTPUT_DIR / f"age_{label}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col="keyword", encoding="utf-8-sig")
            existing[label] = df
            done_sets.append(set(df.index))
        else:
            existing[label] = pd.DataFrame()
            done_sets.append(set())

    # 모든 CSV에 공통으로 존재하는 키워드만 done 처리
    done = done_sets[0].intersection(*done_sets[1:]) if done_sets else set()
    todo = [kw for kw in keywords if kw not in done]
    if not todo:
        return
    print(f"shopping_click_age: {len(todo)}/{len(keywords)} keywords to collect")

    # 키워드별 연령별 트렌드 수집 (중단 시 중간 결과 저장)
    rows = {g["label"]: {} for g in GROUPS}
    try:
        for i, kw in enumerate(todo, 1):
            grouped = fetch_keyword_age(kw)
            for g in GROUPS:
                rows[g["label"]][kw] = grouped.get(g["code"], {})
            n_days = max((len(v) for v in grouped.values()), default=0)
            print(f"[{i}/{len(todo)}] {kw}: {n_days}d")
            time.sleep(SLEEP_SEC)
    finally:
        for g in GROUPS:
            label = g["label"]
            if not rows[label]:
                continue
            df_new = pd.DataFrame.from_dict(rows[label], orient="index")
            df_new = df_new.reindex(list(rows[label].keys()))
            df = pd.concat([existing[label], df_new])
            df.index.name = "keyword"
            df = df.reindex(sorted(df.columns), axis=1)
            path = OUTPUT_DIR / f"age_{label}.csv"
            df.to_csv(path, encoding="utf-8-sig")
            print(f"saved: {path.name} ({df.shape[0]}x{df.shape[1]})")


if __name__ == "__main__":
    main()
