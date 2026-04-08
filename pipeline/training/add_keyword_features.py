"""
키워드별 peak_time 통계 피처를 splits에 추가하는 스크립트

train split에서만 통계를 계산하고 valid/test에 join해 leakage를 방지함

추가 피처 (5개):
  keyword_peak_time_mean        : 키워드별 peak_time 전체 평균
  keyword_peak_time_median      : 키워드별 peak_time 전체 중앙값
  keyword_peak_time_std         : 키워드별 peak_time 전체 표준편차
  keyword_peak_time_recent30    : 키워드별 최근 30일 peak_time 평균 (시간 가중)
  keyword_peak_time_ewm         : 키워드별 지수 가중 평균 (span=30, 최근일수록 가중치 높음)

사용법:
    python pipeline/training/add_keyword_features.py
"""

import os
import pandas as pd

SPLITS_DIR = "data_processed/splits"


def main():
    train = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    valid = pd.read_csv(os.path.join(SPLITS_DIR, "valid.csv"))
    test  = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))

    print(f"train: {len(train):,}행  valid: {len(valid):,}행  test: {len(test):,}행")

    train = train.sort_values(["keyword", "date"]).reset_index(drop=True)

    # 전체 통계
    stats = (
        train.groupby("keyword")["peak_time"]
        .agg(
            keyword_peak_time_mean="mean",
            keyword_peak_time_median="median",
            keyword_peak_time_std="std",
        )
        .reset_index()
    )

    # 최근 30개 관측값 평균 (키워드별 마지막 30행)
    recent30 = (
        train.groupby("keyword")
        .tail(30)
        .groupby("keyword")["peak_time"]
        .mean()
        .rename("keyword_peak_time_recent30")
        .reset_index()
    )

    # 지수 가중 평균: 키워드별로 EWM 계산 후 마지막 값 추출 (train 기준)
    ewm = (
        train.groupby("keyword")["peak_time"]
        .apply(lambda x: x.ewm(span=30, adjust=True).mean().iloc[-1])
        .rename("keyword_peak_time_ewm")
        .reset_index()
    )

    stats = stats.merge(recent30, on="keyword", how="left")
    stats = stats.merge(ewm, on="keyword", how="left")

    global_mean   = train["peak_time"].mean()
    global_median = train["peak_time"].median()
    global_std    = train["peak_time"].std()

    print(f"전체 평균: {global_mean:.3f}  중앙값: {global_median:.1f}  std: {global_std:.3f}")
    print(f"키워드 수: {len(stats)}")

    for name, df in [("train", train), ("valid", valid), ("test", test)]:
        # 기존 keyword feature 컬럼이 있으면 제거 후 재계산
        drop_cols = [c for c in df.columns if c.startswith("keyword_peak_time")]
        df = df.drop(columns=drop_cols, errors="ignore")

        df = df.merge(stats, on="keyword", how="left")
        df["keyword_peak_time_mean"]    = df["keyword_peak_time_mean"].fillna(global_mean)
        df["keyword_peak_time_median"]  = df["keyword_peak_time_median"].fillna(global_median)
        df["keyword_peak_time_std"]     = df["keyword_peak_time_std"].fillna(global_std)
        df["keyword_peak_time_recent30"] = df["keyword_peak_time_recent30"].fillna(global_mean)
        df["keyword_peak_time_ewm"]     = df["keyword_peak_time_ewm"].fillna(global_mean)

        path = os.path.join(SPLITS_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  저장: {path}  ({len(df):,}행 × {len(df.columns)}열)")

    print("완료")


if __name__ == "__main__":
    main()
