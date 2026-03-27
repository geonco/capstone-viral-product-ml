"""
filter_date_range.py
--------------------
sometrend_mention_long.csv에서 수집 기간(2022-03-01 ~ 2026-02-28) 외 행 제거

입력: data/interim/sometrend_mention_long.csv
출력: data/interim/sometrend_mention_long.csv (덮어쓰기)

사용법
  - 클릭 실행: (클릭해서실행)filter_date_range.bat 더블클릭
"""

from pathlib import Path
import pandas as pd

START_DATE = "2022-03-01"
END_DATE   = "2026-02-28"

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH   = PROJECT_ROOT / "data" / "interim" / "sometrend_mention_long.csv"

def main():
    print(f"읽는 중: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

    df["date"] = pd.to_datetime(df["date"])
    before = len(df)

    df = df[(df["date"] >= START_DATE) & (df["date"] <= END_DATE)]
    after = len(df)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(INPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"완료: {before:,}행 → {after:,}행 ({before - after:,}행 제거)")
    print(f"기간: {START_DATE} ~ {END_DATE}")

if __name__ == "__main__":
    main()
