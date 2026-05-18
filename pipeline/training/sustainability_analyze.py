# sustainability 분포 분석 — threshold 설정 및 전략 결정용
# 실행: python -m pipeline.training.sustainability_analyze

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import ROOT, DATE_COL, TRAIN_RATIO, VALID_RATIO, GAP_DAYS, FEAT_COLS
from pipeline.training.data_loader import load_csv
from pipeline.training.train_all import split_data

warnings.filterwarnings("ignore")

TARGETS = ["sustainability_5d", "sustainability_10d", "sustainability_15d"]

# staged 전략 판단 기준
STAGE_THRESH_CANDIDATES = [
    [0.7, 1.3],
    [0.6, 1.0, 1.5],
    [0.5, 1.0],
    [0.8, 1.2],
]

# stability/decay 관련 피처 — sustainability와 상관 높을 것으로 예상
STABILITY_FEATS = [f for f in FEAT_COLS if any(k in f for k in [
    "cv_", "std_", "damping", "drawdown", "direction_streak",
    "reversal", "zero_crossings", "slope_change", "cv_change",
    "past_conversion_rate", "conv_trend", "multi_accel",
])]


def _percentiles(arr):
    ps = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    return {f"p{p}": round(float(np.percentile(arr, p)), 4) for p in ps}


def _bucket_dist(arr, thresholds, names):
    buckets = np.digitize(arr, thresholds, right=True)
    n = len(arr)
    return {names[i]: f"{(buckets==i).sum()}  ({(buckets==i).sum()/n*100:.1f}%)" for i in range(len(names))}


def analyze_distribution(df_train, df_full):
    print("\n" + "=" * 60)
    print("  sustainability 분포 분석")
    print("=" * 60)

    for target in TARGETS:
        arr_full  = df_full[target].dropna().values
        arr_train = df_train[target].dropna().values

        print(f"\n{'─'*55}")
        print(f"  {target}  (전체 {len(arr_full):,}  /  train {len(arr_train):,})")
        print(f"{'─'*55}")

        # 기초 통계
        print(f"  mean={arr_train.mean():.4f}  std={arr_train.std():.4f}"
              f"  skew={float(stats.skew(arr_train)):.4f}  kurt={float(stats.kurtosis(arr_train)):.4f}")

        # 분위수
        pct = _percentiles(arr_train)
        print("  percentiles (train):")
        print("   " + "  ".join(f"{k}={v}" for k, v in pct.items()))

        # 1.0 기준 구간 비율 (declining / stable)
        below1 = (arr_train < 1.0).mean() * 100
        above1 = (arr_train >= 1.0).mean() * 100
        print(f"  < 1.0 (declining): {below1:.1f}%   >= 1.0 (sustaining): {above1:.1f}%")

        # log1p 변환 후 분포 비교
        arr_log = np.log1p(arr_train)
        print(f"  log1p → mean={arr_log.mean():.4f}  std={arr_log.std():.4f}"
              f"  skew={float(stats.skew(arr_log)):.4f}")

        # 후보 threshold별 구간 분포
        print(f"\n  threshold 후보별 구간 분포 (train 기준):")
        for thresholds in STAGE_THRESH_CANDIDATES:
            n_class = len(thresholds) + 1
            names   = [f"b{i}" for i in range(n_class)]
            dist    = _bucket_dist(arr_train, thresholds, names)
            dist_str = "  ".join(f"{k}={v}" for k, v in dist.items())
            print(f"    {thresholds} → {dist_str}")

        # 데이터 기반 분위수 threshold 후보 (p25/p75, p33/p67)
        p25, p50, p75 = np.percentile(arr_train, [25, 50, 75])
        p33, p67      = np.percentile(arr_train, [33, 67])
        print(f"\n  데이터 기반 threshold 후보:")
        print(f"    [p25, p75] = [{p25:.3f}, {p75:.3f}]")
        print(f"    [p33, p67] = [{p33:.3f}, {p67:.3f}]")
        print(f"    [p25, p50, p75] = [{p25:.3f}, {p50:.3f}, {p75:.3f}]")


def analyze_feature_correlation(df_train):
    print("\n" + "=" * 60)
    print("  stability 관련 피처 상관 — sustainability_5d 기준 top 20")
    print("=" * 60)

    target = "sustainability_5d"
    y = df_train[target]

    corrs = {}
    for feat in STABILITY_FEATS:
        if feat in df_train.columns:
            r, _ = stats.spearmanr(df_train[feat].fillna(0), y)
            corrs[feat] = round(float(r), 4)

    top = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
    for feat, r in top:
        bar = "+" * int(abs(r) * 20) if r > 0 else "-" * int(abs(r) * 20)
        print(f"  {feat:<45} r={r:+.4f}  {bar}")


def analyze_extreme_ratio(df_train):
    # staged 전략 시 소수 클래스 비율 — 가중치/quantile 필요 여부 판단
    print("\n" + "=" * 60)
    print("  staged3 [0.7, 1.3] 기준 구간별 샘플 수")
    print("=" * 60)

    for target in TARGETS:
        arr = df_train[target].values
        b0 = (arr < 0.7).sum()
        b1 = ((arr >= 0.7) & (arr < 1.3)).sum()
        b2 = (arr >= 1.3).sum()
        n  = len(arr)
        print(f"  {target}: declining={b0}({b0/n*100:.1f}%)  "
              f"stable={b1}({b1/n*100:.1f}%)  sustained={b2}({b2/n*100:.1f}%)")


def main():
    data_path = str(ROOT / "data_processed" / "dataset.csv")
    df = load_csv(data_path)
    print(f"shape: {df.shape}")

    print(f"\n시계열 분할 (gap={GAP_DAYS}일)")
    train_df, valid_df, test_df = split_data(df, train_stride=1)

    analyze_distribution(train_df, df)
    analyze_feature_correlation(train_df)
    analyze_extreme_ratio(train_df)

    print("\n" + "=" * 60)
    print("  전략 판단 가이드")
    print("=" * 60)
    print("  skew > 1.5        → log 변환 우선 시도")
    print("  특정 구간 < 15%   → staged 전환 시 EXTREME_WEIGHT 필요")
    print("  staged 구간 균등  → hard=False (soft routing)")
    print("  declining 경계 명확 → hard=True 고려")


if __name__ == "__main__":
    main()
