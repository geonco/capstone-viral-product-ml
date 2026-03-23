"""
네이버 데이터랩 검색어 트렌드 API를 사용하여 키워드별 일간 검색 트렌드를 수집합니다.

환경변수 필요:
  NAVER_CLIENT_ID
  NAVER_CLIENT_SECRET

결과: data_raw/search_trend_top5.csv
"""

import os
import requests
import pandas as pd
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

API_URL = "https://openapi.naver.com/v1/datalab/search"

# 상위 5개 키워드 (naver_datalab_top500_keywords.csv 기준)
KEYWORDS = ["두쫀쿠", "촉촉한황치즈칩", "샤오커오라사탕", "두바이찰떡파이", "이클립스"]

# 1년 기간
START_DATE = "2025-03-23"
END_DATE = "2026-03-22"


def fetch_search_trend(keywords, start_date, end_date):
    """Naver DataLab API로 키워드별 일간 검색 트렌드를 가져옵니다."""
    client_id = os.environ["NAVER_CLIENT_ID"]
    client_secret = os.environ["NAVER_CLIENT_SECRET"]

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json",
    }

    body = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": "date",
        "keywordGroups": [
            {"groupName": kw, "keywords": [kw]} for kw in keywords
        ],
    }

    response = requests.post(API_URL, headers=headers, json=body)
    response.raise_for_status()
    return response.json()


def parse_to_dataframe(api_response):
    """API 응답을 DataFrame으로 변환합니다."""
    rows = []
    for result in api_response["results"]:
        keyword = result["title"]
        for point in result["data"]:
            rows.append({
                "date": point["period"],
                "keyword": keyword,
                "ratio": point["ratio"],
            })
    return pd.DataFrame(rows)


def main():
    print(f"키워드: {KEYWORDS}")
    print(f"기간: {START_DATE} ~ {END_DATE}")

    data = fetch_search_trend(KEYWORDS, START_DATE, END_DATE)
    df = parse_to_dataframe(data)

    print(f"\n수집 결과: {len(df)}행 (키워드 {df['keyword'].nunique()}개 × ~{len(df) // df['keyword'].nunique()}일)")

    # 저장
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data_raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "search_trend_top5.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {output_path}")

    # 키워드별 요약
    print("\n키워드별 요약:")
    for kw in KEYWORDS:
        kw_data = df[df["keyword"] == kw]["ratio"]
        print(f"  {kw}: mean={kw_data.mean():.1f}, max={kw_data.max():.1f}, min={kw_data.min():.1f}")


if __name__ == "__main__":
    main()
