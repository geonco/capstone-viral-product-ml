"""
50개 샘플 키워드 피처 + 타겟 생성

입력:
  - test/data_raw/search_trend_sample50.csv          (Naver DataLab 수집 데이터)
  - test/data_raw/keywords_selected_500_seed287977840_sample50_13.csv  (키워드 메타데이터)

출력:
  - test/data_processed/dataset_sample50.csv

피처 (18개):
  [Group A - 검색 모멘텀, 10개]
    search_velocity_3d, search_velocity_7d, search_velocity_14d
    search_acceleration
    ma_ratio_3_14, ma_ratio_7_30
    search_volatility_3d, search_volatility_7d, search_volatility_14d
    trend_slope_30d

  [Group D - 외부 컨텍스트, 4개]
    day_of_week, month, is_holiday, competing_trends_count

  [정적 키워드 피처 - 4개]
    stratum_code, appearance, best_rank, rank_std

타겟 (3개):
  virality_score   — (향후 30일 최대값 − 90일 기준선 평균) / (90일 기준선 표준편차 + ε)
  peak_timing      — 향후 30일 중 피크 도달 일수 (1~30)
  is_viral         — virality_score > 2.0 이면 1, 아니면 0
"""

import os
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit


# ────────────────────────────────────────────────
# 한국 공휴일 (2022~2026)
# ────────────────────────────────────────────────
_KOREAN_HOLIDAYS = {
    # 2022
    "2022-01-01",                                      # 신정
    "2022-02-01", "2022-02-02", "2022-02-03",          # 설날 연휴
    "2022-03-01",                                      # 삼일절
    "2022-05-05", "2022-05-08",                        # 어린이날, 부처님오신날
    "2022-06-01",                                      # 전국동시지방선거
    "2022-06-06",                                      # 현충일
    "2022-08-15",                                      # 광복절
    "2022-09-09", "2022-09-10", "2022-09-11", "2022-09-12",  # 추석 연휴 + 대체
    "2022-10-03", "2022-10-09", "2022-10-10",          # 개천절, 한글날, 대체
    "2022-12-25",                                      # 크리스마스
    # 2023
    "2023-01-01",                                      # 신정
    "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",  # 설날 연휴 + 대체
    "2023-03-01",                                      # 삼일절
    "2023-05-05", "2023-05-27", "2023-05-29",          # 어린이날, 부처님오신날, 대체
    "2023-06-06",                                      # 현충일
    "2023-08-15",                                      # 광복절
    "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-02",  # 추석 연휴 + 대체
    "2023-10-03", "2023-10-09",                        # 개천절, 한글날
    "2023-12-25",                                      # 크리스마스
    # 2024
    "2024-01-01",                                      # 신정
    "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12",  # 설날 연휴 + 대체
    "2024-03-01",                                      # 삼일절
    "2024-04-10",                                      # 국회의원선거
    "2024-05-05", "2024-05-06", "2024-05-15",          # 어린이날, 대체, 부처님오신날
    "2024-06-06",                                      # 현충일
    "2024-08-15",                                      # 광복절
    "2024-09-16", "2024-09-17", "2024-09-18",          # 추석 연휴
    "2024-10-03", "2024-10-09",                        # 개천절, 한글날
    "2024-12-25",                                      # 크리스마스
    # 2025
    "2025-01-01",                                      # 신정
    "2025-01-28", "2025-01-29", "2025-01-30",          # 설날 연휴
    "2025-03-01", "2025-03-03",                        # 삼일절, 대체
    "2025-05-05",                                      # 어린이날
    "2025-05-06",                                      # 부처님오신날 (대체 포함)
    "2025-06-06",                                      # 현충일
    "2025-08-15",                                      # 광복절
    "2025-10-03", "2025-10-05", "2025-10-06", "2025-10-07", "2025-10-08", "2025-10-09",
    "2025-12-25",                                      # 크리스마스
    # 2026
    "2026-01-01",                                      # 신정
    "2026-02-17", "2026-02-18", "2026-02-19",          # 설날 연휴
    "2026-03-01",                                      # 삼일절
}


def is_holiday(date: pd.Timestamp) -> int:
    return int(date.strftime("%Y-%m-%d") in _KOREAN_HOLIDAYS)


# ────────────────────────────────────────────────
# 날짜 채우기
# ────────────────────────────────────────────────
def fill_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """키워드별로 전체 날짜 범위를 채우고 빈 날짜는 ratio=0으로 채웁니다."""
    date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    filled = []
    for keyword, kw_df in df.groupby("keyword"):
        kw_df = kw_df.set_index("date").reindex(date_range)
        kw_df["ratio"] = kw_df["ratio"].fillna(0)
        kw_df["keyword"] = keyword
        kw_df.index.name = "date"
        filled.append(kw_df.reset_index())
    return pd.concat(filled, ignore_index=True)


# ────────────────────────────────────────────────
# Group A: 검색 모멘텀 피처 (10개)
# ────────────────────────────────────────────────
def compute_group_a(group: pd.DataFrame) -> pd.DataFrame:
    """
    키워드 하나의 시계열에 대해 Group A 피처 10개를 계산합니다.
    group은 date 오름차순으로 정렬되어 있어야 합니다.
    """
    s = group["ratio"].reset_index(drop=True)

    # ── A1. search_velocity (3d / 7d / 14d) ──────────────────────────────
    # (ratio_today − ratio_N일전) / (ratio_N일전 + 1)
    velocity_3d  = (s - s.shift(3))  / (s.shift(3)  + 1)
    velocity_7d  = (s - s.shift(7))  / (s.shift(7)  + 1)
    velocity_14d = (s - s.shift(14)) / (s.shift(14) + 1)

    # ── A2. search_acceleration ───────────────────────────────────────────
    # velocity_7d의 1일 변화율 (오늘 velocity − 어제 velocity)
    acceleration = velocity_7d - velocity_7d.shift(1)

    # ── A3. ma_ratio ─────────────────────────────────────────────────────
    ma3  = s.rolling(3,  min_periods=3).mean()
    ma7  = s.rolling(7,  min_periods=7).mean()
    ma14 = s.rolling(14, min_periods=14).mean()
    ma30 = s.rolling(30, min_periods=30).mean()

    ma_ratio_3_14 = ma3  / (ma14 + 1)   # 초단기 vs 중기
    ma_ratio_7_30 = ma7  / (ma30 + 1)   # 단기 vs 장기

    # ── A4. search_volatility (3d / 7d / 14d) ────────────────────────────
    volatility_3d  = s.rolling(3,  min_periods=3).std()
    volatility_7d  = s.rolling(7,  min_periods=7).std()
    volatility_14d = s.rolling(14, min_periods=14).std()

    # ── A5. trend_slope_30d ───────────────────────────────────────────────
    ratio_arr = s.values
    slope = np.full(len(ratio_arr), np.nan)
    for i in range(29, len(ratio_arr)):
        y = ratio_arr[i - 29 : i + 1]
        x = np.arange(30)
        coeffs = polyfit(x, y, 1)
        slope[i] = coeffs[1]  # polyfit returns [intercept, slope]

    group = group.copy()
    group["search_velocity_3d"]   = velocity_3d.values
    group["search_velocity_7d"]   = velocity_7d.values
    group["search_velocity_14d"]  = velocity_14d.values
    group["search_acceleration"]  = acceleration.values
    group["ma_ratio_3_14"]        = ma_ratio_3_14.values
    group["ma_ratio_7_30"]        = ma_ratio_7_30.values
    group["search_volatility_3d"] = volatility_3d.values
    group["search_volatility_7d"] = volatility_7d.values
    group["search_volatility_14d"]= volatility_14d.values
    group["trend_slope_30d"]      = slope
    return group


# ────────────────────────────────────────────────
# 타겟 변수 계산
# ────────────────────────────────────────────────
def compute_targets(group: pd.DataFrame) -> pd.DataFrame:
    """
    virality_score = (향후 30일 최대값 − 90일 기준선 평균) / (90일 기준선 표준편차 + ε)
    peak_timing    = 향후 30일 중 argmax + 1  (1~30)
    is_viral       = virality_score > 2.0

    유효 구간: 90일 기준선 확보 후 ~ 마지막 30일 이전
    """
    ratio = group["ratio"].values
    n = len(ratio)
    eps = 1e-6

    virality_score = np.full(n, np.nan)
    peak_timing    = np.full(n, np.nan)

    # min_denom: 기준선이 모두 0인 희귀 키워드에서 virality_score가
    # 수백만 단위로 폭발하는 것을 방지 (0~100 스케일 기준 최소 std = 0.1)
    min_denom = 0.1

    for i in range(90, n - 30):
        baseline      = ratio[i - 90 : i]
        baseline_mean = baseline.mean()
        baseline_std  = baseline.std()

        future      = ratio[i + 1 : i + 31]
        future_peak = future.max()

        denom = max(baseline_std + eps, min_denom)
        virality_score[i] = (future_peak - baseline_mean) / denom
        peak_timing[i]    = float(future.argmax() + 1)

    group = group.copy()
    group["virality_score"] = virality_score
    group["peak_timing"]    = peak_timing
    group["is_viral"]       = (virality_score > 2.0).astype(float)
    group.loc[np.isnan(virality_score), "is_viral"] = np.nan
    return group


# ────────────────────────────────────────────────
# 메인 파이프라인
# ────────────────────────────────────────────────
def main():
    base_dir = os.path.dirname(__file__)
    data_raw_dir  = os.path.join(base_dir, "..", "data_raw")
    data_proc_dir = os.path.join(base_dir, "..", "data_processed")

    # ── 1. 검색 트렌드 데이터 로드 ─────────────────────────────────────────
    trend_path = os.path.join(data_raw_dir, "search_trend_sample50.csv")
    df = pd.read_csv(trend_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"검색 트렌드 로드: {len(df)}행, {df['keyword'].nunique()}키워드")

    # ── 2. 키워드 메타데이터 로드 ──────────────────────────────────────────
    kw_path = os.path.join(
        data_raw_dir,
        "keywords_selected_500_seed287977840_sample50_13.csv",
    )
    kw_meta = pd.read_csv(kw_path)

    # stratum 레이블 → 숫자 코드 (Q1=1, Q2=2, Q3=3, Q4=4)
    stratum_map = {"Q1_rare": 1, "Q2_low": 2, "Q3_mid": 3, "Q4_freq": 4}
    kw_meta["stratum_code"] = kw_meta["stratum"].map(stratum_map)

    # 정적 피처 컬럼만 추출
    static_cols = ["keyword", "stratum_code", "appearance", "best_rank", "rank_std"]
    kw_static = kw_meta[static_cols].set_index("keyword")

    # ── 3. 날짜 채우기 ────────────────────────────────────────────────────
    df = fill_missing_dates(df)
    n_keywords = df["keyword"].nunique()
    n_days     = len(df) // n_keywords
    print(f"날짜 채움 후: {len(df)}행 ({n_keywords}키워드 × {n_days}일)")

    # ── 4. Group A 피처 + 타겟 계산 ──────────────────────────────────────
    print("Group A 피처 + 타겟 계산 중...")
    results = []
    for keyword, group in df.groupby("keyword"):
        group = group.sort_values("date").reset_index(drop=True)
        group = compute_group_a(group)
        group = compute_targets(group)
        results.append(group)
    df = pd.concat(results, ignore_index=True)
    print("완료")

    # ── 5. competing_trends_count 계산 (Group D 보조) ────────────────────
    # 날짜별로 virality_score > 1.5인 다른 키워드 수 카운트 (벡터화)
    print("competing_trends_count 계산 중...")
    viral_mask = df["virality_score"].notna() & (df["virality_score"] > 1.5)
    viral_count_by_date = (
        df[viral_mask]
        .groupby("date")["keyword"]
        .nunique()
        .rename("_viral_count")
    )
    df = df.merge(viral_count_by_date, on="date", how="left")
    df["_viral_count"] = df["_viral_count"].fillna(0).astype(int)

    # 자기 자신이 바이럴인 경우 1 차감
    df["competing_trends_count"] = (
        df["_viral_count"] - viral_mask.astype(int)
    ).clip(lower=0)
    df.drop(columns=["_viral_count"], inplace=True)

    # ── 6. Group D 피처 계산 ─────────────────────────────────────────────
    df["day_of_week"] = df["date"].dt.dayofweek        # 0=월 ~ 6=일
    df["month"]       = df["date"].dt.month             # 1~12
    df["is_holiday"]  = df["date"].apply(is_holiday)   # 0 or 1

    # ── 7. 정적 키워드 피처 병합 ──────────────────────────────────────────
    df = df.join(kw_static, on="keyword")

    # ── 8. 유효 행 필터링 (피처 + 타겟 모두 있는 행) ──────────────────────
    GROUP_A_COLS = [
        "search_velocity_3d", "search_velocity_7d", "search_velocity_14d",
        "search_acceleration",
        "ma_ratio_3_14", "ma_ratio_7_30",
        "search_volatility_3d", "search_volatility_7d", "search_volatility_14d",
        "trend_slope_30d",
    ]
    GROUP_D_COLS = ["day_of_week", "month", "is_holiday", "competing_trends_count"]
    STATIC_COLS  = ["stratum_code", "appearance", "best_rank", "rank_std"]
    TARGET_COLS  = ["virality_score", "peak_timing", "is_viral"]

    all_feature_cols = GROUP_A_COLS + GROUP_D_COLS + STATIC_COLS

    df_valid = df.dropna(subset=all_feature_cols + TARGET_COLS)
    print(f"유효 데이터: {len(df_valid)}행 / {len(df)}행 전체")

    # ── 9. 저장 ──────────────────────────────────────────────────────────
    os.makedirs(data_proc_dir, exist_ok=True)
    output_path = os.path.join(data_proc_dir, "dataset_sample50.csv")

    output_cols = (
        ["date", "keyword", "ratio"]
        + GROUP_A_COLS
        + GROUP_D_COLS
        + STATIC_COLS
        + TARGET_COLS
    )
    df_valid[output_cols].to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {output_path}")
    print(f"컬럼: {len(output_cols) - 3}개 피처 + 3개 타겟  (date/keyword/ratio 제외)")

    # ── 10. 요약 출력 ─────────────────────────────────────────────────────
    print("\n키워드별 요약:")
    summary = (
        df_valid.groupby("keyword")
        .agg(
            rows=("virality_score", "count"),
            vs_mean=("virality_score", "mean"),
            vs_max=("virality_score", "max"),
            viral_rows=("is_viral", "sum"),
        )
        .sort_values("vs_max", ascending=False)
    )
    for kw, row in summary.iterrows():
        print(
            f"  {kw}: {int(row.rows)}행, "
            f"vs_mean={row.vs_mean:.2f}, vs_max={row.vs_max:.2f}, "
            f"viral_rows={int(row.viral_rows)}"
        )

    # 전체 is_viral 분포
    n_total  = len(df_valid)
    n_viral  = int(df_valid["is_viral"].sum())
    print(f"\n바이럴 비율: {n_viral}/{n_total} = {n_viral/n_total*100:.1f}%")
    print(f"virality_score 분포: "
          f"mean={df_valid['virality_score'].mean():.2f}, "
          f"median={df_valid['virality_score'].median():.2f}, "
          f"max={df_valid['virality_score'].max():.2f}")


if __name__ == "__main__":
    main()
