"""
네이버 데이터랩 검색어 트렌드 - 중위권(300-305위) 6개 키워드 수집

결과: test/data_raw/search_trend_mid6.csv
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

API_URL = "https://openapi.naver.com/v1/datalab/search"

KEYWORDS = ["씬피넛버터샌드", "스쿱마켓", "리얼생감자스틱", "초코룹스", "호두과자답례품", "부샤드다크초콜릿"]

START_DATE = "2025-03-23"
END_DATE = "2026-03-22"


def fetch_search_trend(keywords, start_date, end_date):
    client_id = os.environ["NAVER_CLIENT_ID"]
    client_secret = os.environ["NAVER_CLIENT_SECRET"]

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json",
    }

    # API 최대 5그룹/요청 → 배치 분할
    all_results = []
    for i in range(0, len(keywords), 5):
        batch = keywords[i:i + 5]
        body = {
            "startDate": start_date,
            "endDate": end_date,
            "timeUnit": "date",
            "keywordGroups": [
                {"groupName": kw, "keywords": [kw]} for kw in batch
            ],
        }
        response = requests.post(API_URL, headers=headers, json=body)
        response.raise_for_status()
        all_results.extend(response.json()["results"])
        print(f"  배치 {i // 5 + 1}: {batch}")

    return all_results


def main():
    print(f"키워드: {KEYWORDS}")
    print(f"기간: {START_DATE} ~ {END_DATE}\n")

    results = fetch_search_trend(KEYWORDS, START_DATE, END_DATE)

    rows = []
    for result in results:
        for point in result["data"]:
            rows.append({"date": point["period"], "keyword": result["title"], "ratio": point["ratio"]})
    df = pd.DataFrame(rows)

    print(f"\n수집 결과: {len(df)}행 (키워드 {df['keyword'].nunique()}개)")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data_raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "search_trend_mid6.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {output_path}")

    print("\n키워드별 요약:")
    for kw in KEYWORDS:
        kw_data = df[df["keyword"] == kw]["ratio"]
        if len(kw_data) > 0:
            print(f"  {kw}: {len(kw_data)}일, mean={kw_data.mean():.1f}, max={kw_data.max():.1f}")
        else:
            print(f"  {kw}: 데이터 없음")


if __name__ == "__main__":
    main()
