"""
검색 트렌드 원본 데이터에서 피처와 타겟 변수를 생성합니다.

입력: data_raw/search_trend_top5.csv
출력: data_processed/test_dataset_top5.csv

피처 (Group A - 검색 모멘텀):
  search_velocity_7d, search_acceleration, ma_ratio,
  search_volatility_7d, trend_slope_30d

타겟:
  virality_score  — (future_peak - baseline_90d) / (std_90d + ε)
  peak_timing     — argmax(future 30일) + 1
"""

import os
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit


def fill_missing_dates(df):
    """키워드별로 전체 날짜 범위를 채우고, 빈 날짜는 ratio=0으로 채웁니다."""
    date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    filled = []
    for keyword in df["keyword"].unique():
        kw_df = df[df["keyword"] == keyword].set_index("date")
        kw_df = kw_df.reindex(date_range)
        kw_df["ratio"] = kw_df["ratio"].fillna(0)
        kw_df["keyword"] = keyword
        kw_df.index.name = "date"
        filled.append(kw_df.reset_index())
    return pd.concat(filled, ignore_index=True)


def compute_features(group):
    """키워드 하나의 시계열에 대해 Group A 피처를 계산합니다."""
    ratio = group["ratio"].values

    # search_velocity_7d: 7일 전 대비 변화율
    ratio_7d_ago = group["ratio"].shift(7).values
    velocity = (ratio - ratio_7d_ago) / (ratio_7d_ago + 1)

    # search_acceleration: velocity의 1일 변화
    acceleration = np.diff(velocity, prepend=np.nan)

    # ma_ratio: 7일 이동평균 / 30일 이동평균
    ma7 = group["ratio"].rolling(7, min_periods=7).mean().values
    ma30 = group["ratio"].rolling(30, min_periods=30).mean().values
    ma_ratio = ma7 / (ma30 + 1)

    # search_volatility_7d: 최근 7일 표준편차
    volatility = group["ratio"].rolling(7, min_periods=7).std().values

    # trend_slope_30d: 최근 30일 선형회귀 기울기
    slope = np.full(len(ratio), np.nan)
    for i in range(29, len(ratio)):
        y = ratio[i - 29 : i + 1]
        x = np.arange(30)
        coeffs = polyfit(x, y, 1)
        slope[i] = coeffs[1]  # polyfit returns [intercept, slope]

    group = group.copy()
    group["search_velocity_7d"] = velocity
    group["search_acceleration"] = acceleration
    group["ma_ratio"] = ma_ratio
    group["search_volatility_7d"] = volatility
    group["trend_slope_30d"] = slope
    return group


def compute_targets(group):
    """키워드 하나의 시계열에 대해 타겟 변수를 계산합니다."""
    ratio = group["ratio"].values
    n = len(ratio)
    eps = 1e-6

    virality_score = np.full(n, np.nan)
    peak_timing = np.full(n, np.nan)

    for i in range(90, n - 30):
        baseline = ratio[i - 90 : i]
        baseline_mean = baseline.mean()
        baseline_std = baseline.std()

        future = ratio[i + 1 : i + 31]
        future_peak = future.max()

        virality_score[i] = (future_peak - baseline_mean) / (baseline_std + eps)
        peak_timing[i] = future.argmax() + 1

    group = group.copy()
    group["virality_score"] = virality_score
    group["peak_timing"] = peak_timing
    return group


def main():
    # 데이터 로드
    base_dir = os.path.dirname(__file__)
    raw_path = os.path.join(base_dir, "..", "data_raw", "search_trend_top5.csv")
    df = pd.read_csv(raw_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"원본 데이터: {len(df)}행")

    # 빈 날짜 채우기
    df = fill_missing_dates(df)
    print(f"날짜 채움 후: {len(df)}행 ({df['keyword'].nunique()}키워드 × {len(df) // df['keyword'].nunique()}일)")

    # 피처 생성
    print("피처 생성 중...")
    results = []
    for keyword, group in df.groupby("keyword"):
        group = compute_features(group)
        group = compute_targets(group)
        results.append(group)
    df = pd.concat(results, ignore_index=True)
    print("타겟 생성 완료")

    # 유효한 행만 필터 (피처 + 타겟 모두 있는 행)
    feature_cols = ["search_velocity_7d", "search_acceleration", "ma_ratio",
                    "search_volatility_7d", "trend_slope_30d"]
    target_cols = ["virality_score", "peak_timing"]
    all_cols = feature_cols + target_cols

    df_valid = df.dropna(subset=all_cols)
    print(f"유효 데이터: {len(df_valid)}행")

    # 키워드별 요약
    print("\n키워드별 요약:")
    for kw in df["keyword"].unique():
        kw_data = df_valid[df_valid["keyword"] == kw]
        if len(kw_data) > 0:
            vs = kw_data["virality_score"]
            print(f"  {kw}: {len(kw_data)}행, virality_score mean={vs.mean():.2f}, max={vs.max():.2f}")
        else:
            print(f"  {kw}: 데이터 부족 (90일 baseline + 30일 future 필요)")

    # 저장
    output_dir = os.path.join(base_dir, "..", "data_processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "test_dataset_top5.csv")

    cols = ["date", "keyword", "ratio"] + feature_cols + target_cols
    df_valid[cols].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    main()
