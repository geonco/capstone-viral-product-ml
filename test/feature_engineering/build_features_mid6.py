"""
중위권(300-305위) 키워드 피처 + 타겟 생성

입력: test/data_raw/search_trend_mid6.csv
출력: test/data_processed/test_dataset_mid6.csv
"""

import os
import sys

# 기존 build_features.py의 함수 재활용
sys.path.insert(0, os.path.dirname(__file__))
from build_features import fill_missing_dates, compute_features, compute_targets

import numpy as np
import pandas as pd


def main():
    base_dir = os.path.dirname(__file__)
    raw_path = os.path.join(base_dir, "..", "data_raw", "search_trend_mid6.csv")
    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"원본 데이터: {len(df)}행")

    df = fill_missing_dates(df)
    print(f"날짜 채움 후: {len(df)}행 ({df['keyword'].nunique()}키워드 × {len(df) // df['keyword'].nunique()}일)")

    print("피처 + 타겟 생성 중...")
    results = []
    for keyword, group in df.groupby("keyword"):
        group = compute_features(group)
        group = compute_targets(group)
        results.append(group)
    df = pd.concat(results, ignore_index=True)

    feature_cols = ["search_velocity_7d", "search_acceleration", "ma_ratio",
                    "search_volatility_7d", "trend_slope_30d"]
    target_cols = ["virality_score", "peak_timing"]
    df_valid = df.dropna(subset=feature_cols + target_cols)
    print(f"유효 데이터: {len(df_valid)}행")

    print("\n키워드별 요약:")
    for kw in df["keyword"].unique():
        kw_data = df_valid[df_valid["keyword"] == kw]
        if len(kw_data) > 0:
            vs = kw_data["virality_score"]
            print(f"  {kw}: {len(kw_data)}행, virality mean={vs.mean():.2f}, median={vs.median():.2f}, max={vs.max():.2f}")
        else:
            print(f"  {kw}: 데이터 부족")

    output_dir = os.path.join(base_dir, "..", "data_processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_dataset_mid6.csv")
    cols = ["date", "keyword", "ratio"] + feature_cols + target_cols
    df_valid[cols].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    main()
