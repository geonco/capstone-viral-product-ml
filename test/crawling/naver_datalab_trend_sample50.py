"""
50개 샘플 키워드 검색 트렌드 수집 (2022-01-01 ~ 2026-03-22)

입력: test/data_raw/keywords_selected_500_seed287977840_sample50_13.csv
출력: test/data_raw/search_trend_sample50.csv

환경변수 필요:
  NAVER_CLIENT_ID
  NAVER_CLIENT_SECRET
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

API_URL = "https://openapi.naver.com/v1/datalab/search"
START_DATE = "2022-01-01"
END_DATE = "2026-03-22"
BATCH_SIZE = 5        # API 최대 5 그룹/요청
SLEEP_SECONDS = 1.0   # 배치 간 대기 (rate limit 방지)


def fetch_batch(keywords: list, start_date: str, end_date: str) -> list:
    """배치 단위로 Naver DataLab 검색 트렌드를 가져옵니다."""
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

    rows = []
    for result in response.json().get("results", []):
        keyword = result["title"]
        for point in result["data"]:
            rows.append({
                "date": point["period"],
                "keyword": keyword,
                "ratio": point["ratio"],
            })
    return rows


def main():
    base_dir = os.path.dirname(__file__)

    # 키워드 목록 로드
    kw_path = os.path.join(
        base_dir, "..", "data_raw",
        "keywords_selected_500_seed287977840_sample50_13.csv",
    )
    kw_df = pd.read_csv(kw_path)
    keywords = kw_df["keyword"].tolist()
    print(f"수집 대상 키워드: {len(keywords)}개")
    print(f"수집 기간: {START_DATE} ~ {END_DATE}\n")

    # 배치 단위 API 호출
    all_rows = []
    total_batches = (len(keywords) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(keywords), BATCH_SIZE):
        batch = keywords[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"배치 {batch_num}/{total_batches}: {batch}")

        try:
            rows = fetch_batch(batch, START_DATE, END_DATE)
            all_rows.extend(rows)
            print(f"  → {len(rows)}행 수집")
        except requests.HTTPError as e:
            print(f"  !! API 오류: {e}")
        except KeyError as e:
            print(f"  !! 환경변수 누락: {e} (.env 파일에 NAVER_CLIENT_ID / NAVER_CLIENT_SECRET 설정 필요)")
            break

        if i + BATCH_SIZE < len(keywords):
            time.sleep(SLEEP_SECONDS)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("\n수집된 데이터가 없습니다. API 키 설정을 확인하세요.")
        return

    print(f"\n총 {len(df)}행 수집 (키워드 {df['keyword'].nunique()}개)")

    # 저장
    output_dir = os.path.join(base_dir, "..", "data_raw")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "search_trend_sample50.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {output_path}")

    # 키워드별 요약
    print("\n키워드별 요약:")
    for kw in keywords:
        kw_data = df[df["keyword"] == kw]["ratio"]
        if len(kw_data) > 0:
            nonzero = kw_data[kw_data > 0]
            print(
                f"  {kw}: {len(kw_data)}일, "
                f"비영(非零) {len(nonzero)}일, "
                f"max={kw_data.max():.2f}"
            )
        else:
            print(f"  {kw}: 데이터 없음")


if __name__ == "__main__":
    main()
