"""
build_mention_features.py
-------------------------
sometrend_mention_long.csv로부터 썸트렌드 언급량 Feature 생성

입력: capstone-ml-study-main/sometrend_merged/sometrend_mention_long.csv
출력: capstone-ml-study-main/sometrend_merged/sometrend_mention_features.csv (신규 생성)

Feature 목록 (합계 컬럼 기준)
  mention_ma_3d              최근 3일 이동평균
  mention_ma_7d              최근 7일 이동평균
  mention_growth_3d          최근 3일 증가율
  mention_growth_7d          최근 7일 증가율
  mention_growth_14d         최근 14일 증가율
  mention_acceleration       growth_7d - growth_14d
  mention_std_7d             최근 7일 표준편차
  mention_pos_14d            현재 / 최근 14일 max
  days_since_mention_max_14d 최근 14일 내 peak 이후 경과일

사용법
  - 클릭 실행: (클릭해서실행)build_mention_features.bat 더블클릭
"""

from pathlib import Path
import numpy as np
import pandas as pd

INPUT_PATH  = Path(r"C:\Users\maese\Downloads\capstone-ml-study-main\capstone-ml-study-main\sometrend_merged\sometrend_mention_long.csv")
OUTPUT_PATH = INPUT_PATH.parent / "sometrend_mention_features.csv"


def days_since_max_14d(series: pd.Series) -> pd.Series:
    """최근 14일 내 peak 이후 경과일 계산"""
    arr = series.values
    result = np.empty(len(arr), dtype=float)
    for i in range(len(arr)):
        start  = max(0, i - 13)
        window = arr[start:i + 1]
        argmax = int(np.argmax(window))
        result[i] = len(window) - 1 - argmax
    return pd.Series(result, index=series.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    s = df["합계"].astype(float)

    df["mention_ma_3d"]  = s.rolling(3,  min_periods=1).mean()
    df["mention_ma_7d"]  = s.rolling(7,  min_periods=1).mean()

    eps = 1e-9
    df["mention_growth_3d"]  = (s - s.shift(3))  / (s.shift(3)  + eps)
    df["mention_growth_7d"]  = (s - s.shift(7))  / (s.shift(7)  + eps)
    df["mention_growth_14d"] = (s - s.shift(14)) / (s.shift(14) + eps)

    df["mention_acceleration"] = df["mention_growth_7d"] - df["mention_growth_14d"]

    df["mention_std_7d"]  = s.rolling(7, min_periods=1).std().fillna(0)
    df["mention_pos_14d"] = s / (s.rolling(14, min_periods=1).max() + eps)

    df["days_since_mention_max_14d"] = days_since_max_14d(s)

    return df


def main():
    print(f"읽는 중: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])

    print("Feature 생성 중...")
    result = (
        df.groupby("keyword", sort=False)
          .apply(build_features)
          .reset_index(drop=True)
    )

    result["date"] = result["date"].dt.strftime("%Y-%m-%d")
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"완료: {len(result):,}행")
    print(f"저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
