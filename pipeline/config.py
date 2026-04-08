# 파이프라인 전역 설정

import pandas as pd
from pathlib import Path

# 프로젝트 경로 — raw 데이터, 빌드 결과, 캐시 위치

ROOT       = Path(__file__).resolve().parent.parent
RAW        = ROOT / "data" / "raw"
OUT_PATH   = ROOT / "data" / "processed" / "dataset.csv"
CACHE_DIR  = ROOT / "data" / "cache"
CACHE_META = CACHE_DIR / "meta.json"

PATHS = {
    "search":  RAW / "search" / "absolute.csv",
    "click":   RAW / "shopping_click" / "absolute.csv",
    "mention": RAW / "sometrends" / "sometrend_mention_long.csv",
}

# 신호 채널 — 피처·라벨 산출에 공통 사용하는 채널명과 가중치

SIGNALS        = ["search", "click", "blog", "instagram"]
SIGNAL_WEIGHTS = {"search": 0.4, "click": 0.3, "blog": 0.2, "instagram": 0.1}
MENTION_COLS   = {"블로그": "blog", "인스타그램": "instagram"}  # SomeTrend 컬럼 매핑

# 피처 계산용 윈도우 (일) — 통계량 종류별 최소 데이터 길이가 다름

W_FULL  = [3, 5, 7, 14, 30]   # accel 등 전체 범위에서 사용
W_STAT  = [7, 14, 30]          # skew·kurt·cv·rsi 용, 최소 7일 필요
W_SHORT = [3, 7, 14, 30]       # volume·level·growth·spikiness 용

# 피처 빌드 — build_dataset.py에서 사용하는 시간 범위와 클리핑 기준

START         = pd.Timestamp("2022-05-01")
END           = pd.Timestamp("2026-02-14")
LB            = 60             # lookback 일수
FW            = 15             # forward window 일수
SAMPLE_STRIDE = 1              # 빌드 시 전체, train/valid 서브샘플링은 train.py에서
EPS           = 1e-6           # 0 나누기 방지
CLIP = {
    "level":  (0, 30),
    "growth": (-2, 20),
    "cv":     (0, 10),
    "ratio":  (0, 50),
}

# 학습 — train.py 데이터 분할, 타겟 설정, LightGBM 하이퍼파라미터

DATE_COL    = "date"
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
GAP_DAYS    = 75               # LB + FW, train↔valid↔test 간 누수 방지

# 타겟별 log1p 변환 여부
TARGET_CONFIG = {
    "intensity_5d":        {"log": True},
    "intensity_10d":       {"log": True},
    "intensity_15d":       {"log": True},
    "buzz_composite_5d":   {"log": False},
    "buzz_composite_10d":  {"log": False},
    "buzz_composite_15d":  {"log": False},
    "growth_5d":           {"log": True},
    "growth_10d":          {"log": True},
    "growth_15d":          {"log": True},
    "sustainability_5d":   {"log": False},
    "sustainability_10d":  {"log": False},
    "sustainability_15d":  {"log": False},
    "crash_5d":            {"log": False},
    "crash_10d":           {"log": False},
    "crash_15d":           {"log": False},
}

# LightGBM 기본 하이퍼파라미터
DEFAULT_PARAMS = {
    "boosting_type"    : "gbdt",
    "n_estimators"     : 1000,
    "learning_rate"    : 0.05,
    "num_leaves"       : 64,
    "max_depth"        : -1,
    "min_child_samples": 20,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 0.1,
    "random_state"     : 42,
    "n_jobs"           : -1,
    "verbose"          : -1,
}

# Optuna TPE 탐색 범위 — (타입, min, max [, log 스케일])
SEARCH_SPACE = {
    "learning_rate"    : ("float", 0.01, 0.15,  True),
    "num_leaves"       : ("int",   15,   127),
    "max_depth"        : ("int",   4,    10),
    "min_child_samples": ("int",   50,   500),
    "subsample"        : ("float", 0.6,  0.95,  False),
    "colsample_bytree" : ("float", 0.4,  0.9,   False),
    "reg_alpha"        : ("float", 1e-3, 10.0,  True),
    "reg_lambda"       : ("float", 1e-3, 10.0,  True),
}

# 컬럼 스키마 — FEAT_COLS/LABEL_COLS 생성, data_loader와 build_dataset에서 참조

def _sig_cols(p):
    # 신호 하나의 피처 컬럼명 생성
    c = []

    # 기본 통계 — volume, level, growth, accel, rsi, shape, cv, lag, std, pos 등
    for n in W_SHORT:   c.append(f"{p}_volume_{n}d")
    for n in W_SHORT:   c.append(f"{p}_level_{n}d")
    for n in W_SHORT:   c.append(f"{p}_growth_{n}d")
    c += [f"{p}_accel_short", f"{p}_accel_mid", f"{p}_accel_long"]
    for n in [14, 30]:  c.append(f"{p}_rsi_{n}d")
    for n in [14, 30]:  c.append(f"{p}_skew_{n}d")
    for n in [14, 30]:  c.append(f"{p}_kurt_{n}d")
    for n in W_SHORT:   c.append(f"{p}_spikiness_{n}d")
    for n in W_STAT:    c.append(f"{p}_cv_{n}d")
    c.append(f"{p}_cv_change")
    for n in [1, 3, 7, 14, 30]: c.append(f"{p}_lag_{n}d")
    for n in W_STAT:    c.append(f"{p}_std_{n}d")
    for n in W_FULL:    c.append(f"{p}_pos_{n}d")
    c.append(f"days_since_{p}_max_14d")

    # 추세·파동·레짐 — 기울기, 볼린저 밴드, 방향 전환, 돌파 강도 등
    c += [f"{p}_slope_3d", f"{p}_slope_7d", f"{p}_slope_14d"]
    c += [f"{p}_norm_slope_7d", f"{p}_slope_change"]
    c += [f"{p}_trend_r2_7d", f"{p}_residual_energy_7d", f"{p}_curvature_14d"]
    c += [f"{p}_vol_ratio", f"{p}_range_squeeze", f"{p}_damping"]
    c += [f"{p}_zero_crossings_14d", f"{p}_direction_streak", f"{p}_reversal_3d"]
    c += [f"{p}_band_position", f"{p}_band_width", f"{p}_drawdown_14d", f"{p}_breakout_strength"]
    c.append(f"{p}_band_approach")

    # 일별 모멘텀·변화율 — 전일 수익률, 다이버전스, 최근 7일 개별 변화율
    c += [f"{p}_daily_return_1d", f"{p}_daily_returns_3d_avg"]
    c += [f"{p}_momentum_divergence", f"{p}_daily_accel_1d"]
    for d in range(1, 8):
        c.append(f"{p}_change_{d}d_ago")

    return c


FEAT_COLS = []
for _p in SIGNALS:
    FEAT_COLS += _sig_cols(_p)

# 교차 피처 — 채널 간 ratio, 선행/후행, 교차상관, 기울기 괴리
FEAT_COLS += [
    "click_search_ratio", "click_lead_7d", "click_lead_14d", "click_search_pos_gap",
    "blog_lead_7d", "blog_search_ratio",
    "instagram_lead_7d", "instagram_search_ratio",
]
FEAT_COLS += [
    "blog_search_best_lag", "blog_search_peak_xcorr",
    "insta_search_best_lag", "insta_search_peak_xcorr",
    "click_search_best_lag", "click_search_peak_xcorr",
    "blog_search_dir_agree", "insta_search_dir_agree", "click_search_dir_agree",
    "blog_search_slope_gap", "insta_search_slope_gap",
    "multi_slope_count",
    "blog_search_lead_magnitude", "insta_search_lead_magnitude", "click_search_lead_magnitude",
    "weighted_slope_strength", "activation_spread",
]

# 서지 피처 — blog/instagram의 급등 강도·가속도·선행 일수
FEAT_COLS += [
    "blog_surge_intensity_3d", "blog_surge_intensity_7d",
    "blog_surge_accel_3d", "blog_surge_accel_7d", "blog_surge_lead_days",
    "instagram_surge_intensity_3d", "instagram_surge_intensity_7d",
    "instagram_surge_accel_3d", "instagram_surge_accel_7d", "instagram_surge_lead_days",
]

# 멀티채널 종합 — 전환율, 활성 채널 수, 가중 가속도, buzz z-score
_W_MULTI = [1, 3, 7, 14, 30]
for _n in _W_MULTI:
    FEAT_COLS += [f"past_conversion_rate_{_n}d", f"past_channel_active_{_n}d"]
    if _n >= 3:
        FEAT_COLS.append(f"multi_accel_{_n}d")
    FEAT_COLS.append(f"past_buzz_zscore_{_n}d")
FEAT_COLS += ["conv_trend"]

# 캘린더
FEAT_COLS += ["month"]

# 라벨 — forward window에서 산출하는 예측 타겟
LABEL_COLS = [
    "intensity_5d", "intensity_10d", "intensity_15d",
    "buzz_composite_5d", "buzz_composite_10d", "buzz_composite_15d",
    "growth_5d", "growth_10d", "growth_15d",
    "sustainability_5d", "sustainability_10d", "sustainability_15d",
    "crash_5d", "crash_10d", "crash_15d",
]

META_COLS     = ["keyword", "date"]
ALL_COLS      = META_COLS + FEAT_COLS + LABEL_COLS
REQUIRED_COLS = FEAT_COLS
OPTIONAL_COLS = META_COLS + LABEL_COLS

assert len(FEAT_COLS) == 348, f"expected 348, got {len(FEAT_COLS)}"
