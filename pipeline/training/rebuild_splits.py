"""
새 peak_time 특화 피처를 기존 splits에 추가해 재저장하는 스크립트

dataset_v2.csv의 기존 feature 컬럼(slope, days_since_max)으로부터
새 feature를 계산해 data_processed/splits/ 에 덮어씀

사용법:
    python pipeline/training/rebuild_splits.py
"""

import os

import numpy as np
import pandas as pd


def _add_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """기존 feature 컬럼을 재활용해 새 피처 11개를 추가"""
    df = df.copy()

    # 추세 방향: sign(slope) → +1 상승, -1 하락, 0 보합 (4개)
    df["search_slope_sign_3d"] = np.sign(df["search_slope_3d"])
    df["search_slope_sign_7d"] = np.sign(df["search_slope_7d"])
    df["click_slope_sign_3d"]  = np.sign(df["click_slope_3d"])
    df["click_slope_sign_7d"]  = np.sign(df["click_slope_7d"])

    # 피크 통과 여부: days_since_max > 0이면 이미 피크를 지남 (6개)
    df["search_peak_passed_7d"]  = (df["days_since_search_max_7d"]  > 0).astype(float)
    df["search_peak_passed_14d"] = (df["days_since_search_max_14d"] > 0).astype(float)
    df["click_peak_passed_7d"]   = (df["days_since_click_max_7d"]   > 0).astype(float)
    df["click_peak_passed_14d"]  = (df["days_since_click_max_14d"]  > 0).astype(float)
    df["mention_peak_passed_7d"]  = (df["days_since_mention_max_7d"]  > 0).astype(float)
    df["mention_peak_passed_14d"] = (df["days_since_mention_max_14d"] > 0).astype(float)

    # mention 추세 방향 (2개)
    df["mention_slope_sign_3d"] = np.sign(df["mention_slope_3d"])
    df["mention_slope_sign_7d"] = np.sign(df["mention_slope_7d"])

    return df


def main():
    raw_path   = "data_raw/dataset_v2.csv"
    splits_dir = "data_processed/splits"

    print(f"원본 데이터 로드: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"  {len(df):,}행 × {len(df.columns)}열")

    print("새 피처 계산 중...")
    df = _add_new_features(df)
    print(f"  완료 → {len(df.columns)}열")

    # 시계열 기준 분할 (70/15/15)
    n  = len(df)
    t1 = int(n * 0.70)
    t2 = int(n * 0.85)
    train_df = df.iloc[:t1].copy()
    valid_df = df.iloc[t1:t2].copy()
    test_df  = df.iloc[t2:].copy()

    os.makedirs(splits_dir, exist_ok=True)
    for name, part in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        path = os.path.join(splits_dir, f"{name}.csv")
        part.to_csv(path, index=False)
        print(f"  저장: {path}  ({len(part):,}행)")

    print("완료")


if __name__ == "__main__":
    main()
