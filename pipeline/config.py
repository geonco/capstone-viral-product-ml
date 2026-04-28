# 파이프라인 전역 설정

import pandas as pd
from pathlib import Path

# 프로젝트 경로 — raw 데이터, 빌드 결과, 캐시 위치

ROOT       = Path(__file__).resolve().parent.parent
RAW        = ROOT / "data_raw"
OUT_PATH   = ROOT / "data_processed" / "dataset.csv"
CACHE_DIR  = ROOT / "data_processed" / "cache"
CACHE_META = CACHE_DIR / "meta.json"

PATHS = {
    "search":  RAW / "search" / "absolute.csv",
    "click":   RAW / "shopping_click" / "absolute.csv",
    "mention": RAW / "sometrends" / "sometrend_mention_long.csv",
}

# 신호 채널 — 피처·라벨 산출에 공통 사용하는 채널명과 가중치

SIGNALS        = ["search", "click", "blog", "instagram"]
SIGNAL_WEIGHTS = {"search": 0.4, "click": 0.3, "blog": 0.2, "instagram": 0.1}
MENTION_COLS   = {"블로그": "blog", "인스타그램": "instagram", "커뮤니티": "community"}  # SomeTrend 컬럼 매핑

# 커뮤니티 채널 허용 피처 — 2026-04-13 재검증 결과 상대화 피처만 허용 (drift 위험)
# 금지: community_volume/std/lag/growth/cv/rsi/slope 계열
_W_SHORT = [3, 7, 14, 30]
_W_FULL_COMM = [3, 5, 7, 14, 30]
SIG_COMMUNITY_ALLOW = set(
    [f"community_zscore_{n}d" for n in [3, 7]]
    + [f"community_spikiness_{n}d" for n in _W_SHORT]
    + [f"community_level_{n}d" for n in _W_SHORT]
    + [f"community_pos_{n}d" for n in _W_FULL_COMM]
)

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
    "spike_5d":            {"log": False},
    "spike_10d":           {"log": False},
    "spike_15d":           {"log": False},
}

# LightGBM 기본 하이퍼파라미터
DEFAULT_PARAMS = {
    "boosting_type"    : "gbdt",
    "n_estimators"     : 3000,
    "learning_rate"    : 0.05,
    "num_leaves"       : 64,
    "max_depth"        : -1,
    "min_child_samples": 200,
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

# 피처 마스크 — train.py --mask <이름> 으로 선택 적용

# 관성 마스크 — 3일 집계 관성 피처
MASK_MOMENTUM = (
    [f"{p}_slope_3d" for p in SIGNALS]
    + [f"{p}_daily_returns_3d_avg" for p in SIGNALS]
)

# 자기회귀 마스크 — buzz_composite에서 past_buzz_zscore가 SHAP 38% 독점
MASK_BUZZ_AUTOREGRESSIVE = [f"past_buzz_zscore_{n}d" for n in [1, 3, 7, 14, 30]]

# 노이즈 마스크 — SHAP ≈ 0인 피처 일괄 제거

MASK_DEAD = (
    [f"{p}_momentum_divergence" for p in SIGNALS]
    + [f"{p}_reversal_3d" for p in SIGNALS]
    + [f"{p}_direction_streak" for p in SIGNALS]
)

MASK_WAVE = (
    [f"{p}_vol_ratio" for p in SIGNALS]
    + [f"{p}_range_squeeze" for p in SIGNALS]
    + [f"{p}_damping" for p in SIGNALS]
    + [f"{p}_zero_crossings_14d" for p in SIGNALS]
)

MASK_SURGE = (
    [f"{pp}_surge_{kind}_{n}d"
     for pp in ["blog", "instagram", "community"]
     for kind in ["intensity", "accel"]
     for n in [3, 7]]
    + [f"{pp}_surge_lead_days" for pp in ["blog", "instagram", "community"]]
)

MASK_PCA = [f"past_channel_active_{n}d" for n in [1, 3, 7, 14, 30]]

# MASK_RESIDUAL은 제거 — residual_energy_7d가 수식중복으로 FEAT_COLS에서 완전 삭제됨

MASK_NOISE = MASK_DEAD + MASK_WAVE + MASK_SURGE + MASK_PCA

# 이름 → 피처 목록 매핑 — train.py에서 참조. 삭제된 피처는 FEAT_COLS에 없으므로 교집합 자동
MASK_REGISTRY = {
    "momentum": MASK_MOMENTUM,
    "buzz":     MASK_BUZZ_AUTOREGRESSIVE,
    "noise":    MASK_NOISE,
    "dead":     MASK_DEAD,
    "wave":     MASK_WAVE,
    "surge":    MASK_SURGE,
}

# 컬럼 스키마 — FEAT_COLS/LABEL_COLS 생성, data_loader와 build_dataset에서 참조

def _sig_cols(p):
    # 신호 하나의 피처 컬럼명 생성
    c = []

    # 절대 규모 — raw + log1p 병존. log1p는 skewness 10-125의 heavy tail 완화
    for n in W_SHORT:   c.append(f"{p}_volume_{n}d")
    for n in W_SHORT:   c.append(f"{p}_log_volume_{n}d")

    # 상대 위치 — 전체 평균 대비 최근 윈도우 비율
    for n in W_SHORT:   c.append(f"{p}_level_{n}d")

    # 성장률 — n=30은 level_30d와 |r|=1.00 중복으로 제거
    for n in [3, 7, 14]: c.append(f"{p}_growth_{n}d")
    c += [f"{p}_accel_short", f"{p}_accel_mid", f"{p}_accel_long"]
    for n in [14, 30]:  c.append(f"{p}_rsi_{n}d")
    for n in [14, 30]:  c.append(f"{p}_skew_{n}d")
    for n in [14, 30]:  c.append(f"{p}_kurt_{n}d")
    for n in W_SHORT:   c.append(f"{p}_spikiness_{n}d")
    for n in W_STAT:    c.append(f"{p}_cv_{n}d")
    c.append(f"{p}_cv_change")
    for n in [1, 3, 7, 14, 30]: c.append(f"{p}_lag_{n}d")
    for n in W_STAT:    c.append(f"{p}_std_{n}d")
    for n in W_STAT:    c.append(f"{p}_log_std_{n}d")
    for n in W_FULL:    c.append(f"{p}_pos_{n}d")
    c.append(f"days_since_{p}_max_14d")

    # 추세·파동·레짐 — residual_energy_7d는 trend_r2_7d와 |r|=1.00으로 제거, band_width는 cv_30d와 동일로 제거
    c += [f"{p}_slope_3d", f"{p}_slope_7d", f"{p}_slope_14d"]
    c += [f"{p}_norm_slope_7d", f"{p}_slope_change"]
    c += [f"{p}_trend_r2_7d", f"{p}_curvature_14d"]
    c += [f"{p}_vol_ratio", f"{p}_range_squeeze", f"{p}_damping"]
    c += [f"{p}_zero_crossings_14d", f"{p}_direction_streak", f"{p}_reversal_3d"]
    c += [f"{p}_band_position", f"{p}_drawdown_14d", f"{p}_breakout_strength"]
    c.append(f"{p}_band_approach")

    # 일별 모멘텀·변화율 — change_1d_ago는 daily_return_1d와 동일로 제거
    c += [f"{p}_daily_return_1d", f"{p}_daily_returns_3d_avg"]
    c += [f"{p}_momentum_divergence", f"{p}_daily_accel_1d"]
    for d in range(2, 8):
        c.append(f"{p}_change_{d}d_ago")

    # 비대칭 변동성 — 하락/상승 일별 변화의 표준편차 분리
    c += [f"{p}_downside_std_7d", f"{p}_upside_std_7d"]

    # 개별 시그널 z-score — baseline 대비 최근 이상치 정도
    c += [f"{p}_zscore_3d", f"{p}_zscore_7d"]

    return c


# 2026-04-13 SHAP 기반 클러스터 축소 대상 — Spearman |r|>0.7 묶음에서 SHAP 상위 2개만 유지, 나머지 제거
# 근거: memory/project_plateau_diagnosis.md. 실효 독립 축 51개(PC 80%)
CLUSTER_DROP = set([
    'blog_cv_30d', 'blog_downside_std_7d', 'blog_pos_14d', 'blog_pos_30d', 'blog_pos_3d',
    'blog_pos_5d', 'blog_pos_7d', 'blog_search_slope_gap', 'blog_spikiness_14d',
    'blog_spikiness_30d', 'blog_spikiness_7d', 'blog_std_14d', 'blog_std_30d', 'blog_std_7d',
    'blog_surge_accel_7d', 'blog_upside_std_7d', 'blog_volume_3d', 'blog_volume_7d',
    'click_cv_14d', 'click_downside_std_7d', 'click_lag_3d', 'click_level_30d', 'click_level_7d',
    'click_spikiness_14d', 'click_std_14d', 'click_std_7d', 'click_upside_std_7d',
    'click_volume_14d', 'click_volume_3d', 'click_volume_7d', 'click_zscore_3d', 'click_zscore_7d',
    'instagram_cv_30d', 'instagram_damping', 'instagram_downside_std_7d', 'instagram_growth_3d',
    'instagram_growth_7d', 'instagram_pos_14d', 'instagram_pos_30d', 'instagram_pos_3d',
    'instagram_pos_5d', 'instagram_pos_7d', 'instagram_range_squeeze', 'instagram_search_ratio',
    'instagram_skew_14d', 'instagram_spikiness_14d', 'instagram_spikiness_30d',
    'instagram_std_14d', 'instagram_std_30d', 'instagram_std_7d',
    'instagram_surge_intensity_3d', 'instagram_upside_std_7d', 'instagram_vol_ratio',
    'instagram_volume_14d', 'instagram_volume_30d', 'instagram_volume_3d', 'instagram_volume_7d',
    'instagram_zero_crossings_14d', 'multi_accel_30d', 'past_buzz_zscore_14d',
    'past_conversion_rate_14d', 'past_conversion_rate_30d', 'past_conversion_rate_3d',
    'search_band_position', 'search_breakout_strength', 'search_cv_7d',
    'search_downside_std_7d', 'search_lag_3d', 'search_level_14d', 'search_level_30d',
    'search_level_3d', 'search_level_7d', 'search_rsi_30d', 'search_spikiness_14d',
    'search_spikiness_30d', 'search_spikiness_7d', 'search_std_14d', 'search_std_30d',
    'search_upside_std_7d', 'search_volume_14d', 'search_volume_3d', 'search_volume_7d',
    'search_zscore_3d', 'search_zscore_7d',
])

FEAT_COLS = []
for _p in SIGNALS:
    FEAT_COLS += [c for c in _sig_cols(_p) if c not in CLUSTER_DROP]

# 교차 피처 — 채널 간 ratio, 선행/후행, 교차상관, 기울기 괴리
_cross = [
    "click_search_ratio", "click_lead_7d", "click_lead_14d", "click_search_pos_gap",
    "blog_lead_7d", "blog_search_ratio",
    "instagram_lead_7d", "instagram_search_ratio",
    "blog_search_best_lag", "blog_search_peak_xcorr",
    "insta_search_best_lag", "insta_search_peak_xcorr",
    "click_search_best_lag", "click_search_peak_xcorr",
    "blog_search_dir_agree", "insta_search_dir_agree", "click_search_dir_agree",
    "blog_search_slope_gap", "insta_search_slope_gap",
    "multi_slope_count",
    "blog_search_lead_magnitude", "insta_search_lead_magnitude", "click_search_lead_magnitude",
    "weighted_slope_strength", "activation_spread",
]
FEAT_COLS += [c for c in _cross if c not in CLUSTER_DROP]

# 서지 피처 — blog/instagram + community의 급등 강도·가속도·선행 일수
_surge = [
    "blog_surge_intensity_3d", "blog_surge_intensity_7d",
    "blog_surge_accel_3d", "blog_surge_accel_7d", "blog_surge_lead_days",
    "instagram_surge_intensity_3d", "instagram_surge_intensity_7d",
    "instagram_surge_accel_3d", "instagram_surge_accel_7d", "instagram_surge_lead_days",
    "community_surge_intensity_3d", "community_surge_intensity_7d",
    "community_surge_accel_3d", "community_surge_accel_7d", "community_surge_lead_days",
]
FEAT_COLS += [c for c in _surge if c not in CLUSTER_DROP]

# 멀티채널 종합 — 전환율, 활성 채널 수, 가중 가속도, buzz z-score
_multi = []
for _n in [1, 3, 7, 14, 30]:
    _multi += [f"past_conversion_rate_{_n}d", f"past_channel_active_{_n}d"]
    if _n >= 3:
        _multi.append(f"multi_accel_{_n}d")
    _multi.append(f"past_buzz_zscore_{_n}d")
_multi.append("conv_trend")
FEAT_COLS += [c for c in _multi if c not in CLUSTER_DROP]

# 캘린더 — month는 PSI=7.92로 제거, sin/cos circular encoding으로 대체
FEAT_COLS += ["month_sin", "month_cos"]

# 커뮤니티 채널 sig_feat 피처 — 2026-04-13 재검증 결과 상대화 피처만 허용 (drift 위험 최소화)
# surge 계열은 위 _surge에 이미 포함됨
FEAT_COLS += sorted(SIG_COMMUNITY_ALLOW)

# 라벨 — forward window에서 산출하는 예측 타겟
LABEL_COLS = [
    "buzz_composite_5d", "buzz_composite_10d", "buzz_composite_15d",
    "growth_5d", "growth_10d", "growth_15d",
    "sustainability_5d", "sustainability_10d", "sustainability_15d",
    "crash_5d", "crash_10d", "crash_15d",
    "spike_5d", "spike_10d", "spike_15d",
]

META_COLS     = ["keyword", "date"]
ALL_COLS      = META_COLS + FEAT_COLS + LABEL_COLS
REQUIRED_COLS = FEAT_COLS
OPTIONAL_COLS = META_COLS + LABEL_COLS

# 2026-04-13 Phase 1 피처 개편 — 수식중복 16 제거, log1p 28 추가, month→sin/cos, 클러스터 축소, community 추가
# 중복 검증 (동일 이름이 두 번 들어가지 않는지)
assert len(FEAT_COLS) == len(set(FEAT_COLS)), f"duplicate in FEAT_COLS: {[c for c in FEAT_COLS if FEAT_COLS.count(c) > 1][:5]}"
# 최종 피처 수는 구조 변경에 따라 재빌드 후 확정되므로 범위로 검증
assert 280 <= len(FEAT_COLS) <= 330, f"expected 280-330, got {len(FEAT_COLS)}"


# 실험 7 — 마스킹 조합 실험용 마스크 추가, 카테고리별 정의

def _by_prefix(prefix):
    # 접두어 일치 피처 추출 — 채널·구조 마스크 자동 생성
    return [c for c in FEAT_COLS if c.startswith(prefix)]


# 채널 단위 — 채널 전체를 한 번에 제외
MASK_SEARCH    = _by_prefix("search_")
MASK_CLICK     = _by_prefix("click_")
MASK_BLOG      = _by_prefix("blog_")
MASK_INSTAGRAM = _by_prefix("instagram_")
MASK_COMMUNITY = _by_prefix("community_")

# 변환 단위 — Phase 1에서 추가된 log·calendar 변환 효과 격리 검증
MASK_LOG      = [c for c in FEAT_COLS if "_log_volume_" in c or "_log_std_" in c]
MASK_CALENDAR = ["month_sin", "month_cos"]

# 피처 유형 — 통계량·변동성·도함수 종류별
MASK_VOLUME    = [c for c in FEAT_COLS if "_volume_" in c and "_log_volume_" not in c]
MASK_STD       = [c for c in FEAT_COLS if "_std_" in c and "_log_std_" not in c
                  and "_downside_std_" not in c and "_upside_std_" not in c]
MASK_CV        = [c for c in FEAT_COLS if "_cv_" in c]
MASK_LAG       = [c for c in FEAT_COLS if "_lag_" in c]
MASK_CHANGE    = [c for c in FEAT_COLS if "_change_" in c and c.endswith("_ago")]
MASK_SPIKINESS = [c for c in FEAT_COLS if "_spikiness_" in c]
MASK_POS       = [c for c in FEAT_COLS if "_pos_" in c]
MASK_LEVEL     = [c for c in FEAT_COLS if "_level_" in c]
MASK_GROWTH    = [c for c in FEAT_COLS if "_growth_" in c or "_accel_" in c]
MASK_RSI       = [c for c in FEAT_COLS if "_rsi_" in c]
MASK_SLOPE     = [c for c in FEAT_COLS if "_slope" in c or "_curvature_" in c or "_trend_r2_" in c]
MASK_BAND      = [c for c in FEAT_COLS if "_band_" in c or "_drawdown_" in c or "_breakout_" in c]
MASK_SKEWKURT  = [c for c in FEAT_COLS if "_skew_" in c or "_kurt_" in c]

# 구조 단위 — 채널 간 교차·자기회귀·모멘텀 등
_CROSS_KEYS = ("click_search", "blog_search", "insta_search",
               "blog_lead", "instagram_lead", "click_lead")
_CROSS_EXTRAS = {"multi_slope_count", "weighted_slope_strength", "activation_spread"}
MASK_CROSS = [c for c in FEAT_COLS
              if any(c.startswith(k) for k in _CROSS_KEYS) or c in _CROSS_EXTRAS]
MASK_MULTICHANNEL = [c for c in FEAT_COLS
                     if c.startswith("past_conversion_rate_")
                     or c.startswith("past_channel_active_")
                     or c.startswith("multi_accel_")
                     or c == "conv_trend"]
MASK_ZSCORE = [c for c in FEAT_COLS
               if "_zscore_" in c and not c.startswith("past_buzz_zscore")]
MASK_DAILY_MOMENTUM = [c for c in FEAT_COLS
                       if "_daily_return_" in c or "_daily_returns_" in c or "_daily_accel_" in c]
MASK_AUTOREGRESSIVE = [c for c in FEAT_COLS
                       if c.startswith("past_buzz_zscore_")
                       or c.endswith("_lag_1d") or c.endswith("_lag_3d")]

# 대칭·극단 — 상하한 비대칭, 꼬리 통계
MASK_ASYMMETRY    = [c for c in FEAT_COLS if "_downside_std_" in c or "_upside_std_" in c]
MASK_EXTREMA      = [c for c in FEAT_COLS
                     if "_spikiness_" in c or "_drawdown_" in c or "_breakout_" in c
                     or "_skew_" in c or "_kurt_" in c]
MASK_PEAK_RECENCY = [c for c in FEAT_COLS if c.startswith("days_since_")]

# MASK_REGISTRY 확장 — 기존 6개와 동일 인터페이스로 train.py에서 호출
MASK_REGISTRY.update({
    "search":         MASK_SEARCH,
    "click":          MASK_CLICK,
    "blog":           MASK_BLOG,
    "instagram":      MASK_INSTAGRAM,
    "community":      MASK_COMMUNITY,
    "log":            MASK_LOG,
    "calendar":       MASK_CALENDAR,
    "volume":         MASK_VOLUME,
    "std":            MASK_STD,
    "cv":             MASK_CV,
    "lag":            MASK_LAG,
    "change":         MASK_CHANGE,
    "spikiness":      MASK_SPIKINESS,
    "pos":            MASK_POS,
    "level":          MASK_LEVEL,
    "growth":         MASK_GROWTH,
    "rsi":            MASK_RSI,
    "slope":          MASK_SLOPE,
    "band":           MASK_BAND,
    "skewkurt":       MASK_SKEWKURT,
    "cross":          MASK_CROSS,
    "multichannel":   MASK_MULTICHANNEL,
    "zscore":         MASK_ZSCORE,
    "daily_momentum": MASK_DAILY_MOMENTUM,
    "autoregressive": MASK_AUTOREGRESSIVE,
    "asymmetry":      MASK_ASYMMETRY,
    "extrema":        MASK_EXTREMA,
    "peak_recency":   MASK_PEAK_RECENCY,
})

# 빈 마스크 자동 제거 — CLUSTER_DROP 등으로 피처가 모두 빠진 항목 (asymmetry 등)
_empty_masks = [k for k, v in MASK_REGISTRY.items() if not v]
if _empty_masks:
    print(f"[config] 빈 마스크 제외: {_empty_masks}")
    for k in _empty_masks:
        del MASK_REGISTRY[k]


# 사전 정의 조합 — scripts/mask_experiments.py 일괄 실행, train.py --combo 단건 호출
def _build_combos():
    import itertools as _it
    masks = list(MASK_REGISTRY.keys())
    combos = {"baseline": []}

    # Singleton — 모든 마스크 단독 적용으로 그룹별 효과 격리
    for m in masks:
        combos[f"S_{m}"] = [m]

    # Doubles — buzz·noise·momentum 핵심축과 나머지 마스크 페어링
    for pivot in ("buzz", "noise", "momentum"):
        for m in masks:
            if m == pivot:
                continue
            key = f"D_{'_'.join(sorted([pivot, m]))}"
            combos.setdefault(key, sorted([pivot, m]))

    # Doubles — 채널 간 페어
    channels = ["search", "click", "blog", "instagram", "community"]
    for a, b in _it.combinations(channels, 2):
        combos.setdefault(f"D_{a}_{b}", sorted([a, b]))

    # Doubles — 유형 간 페어
    type_pairs = [("cv", "std"), ("lag", "change"), ("slope", "band"),
                  ("level", "growth"), ("pos", "skewkurt"), ("volume", "std"),
                  ("spikiness", "skewkurt"), ("slope", "growth")]
    for a, b in type_pairs:
        combos.setdefault(f"D_{'_'.join(sorted([a, b]))}", sorted([a, b]))

    # Doubles — 구조 간 페어
    struct_pairs = [("log", "calendar"), ("cross", "multichannel"),
                    ("autoregressive", "zscore"), ("daily_momentum", "rsi"),
                    ("extrema", "asymmetry")]
    for a, b in struct_pairs:
        combos.setdefault(f"D_{'_'.join(sorted([a, b]))}", sorted([a, b]))

    # Triples — buzz+noise+X 로 나머지 마스크 추가 효과 확인
    for m in masks:
        if m in ("buzz", "noise"):
            continue
        combos[f"T_buzz_noise_{m}"] = sorted(["buzz", "noise", m])

    # Triples — 채널 3개 동시 제거
    for tri in _it.combinations(channels, 3):
        combos[f"T_3ch_{'_'.join(tri)}"] = list(tri)

    # Triples — 유형 3개 묶음
    type_triples = [("cv", "std", "band"), ("lag", "change", "daily_momentum"),
                    ("slope", "growth", "rsi"), ("pos", "level", "skewkurt"),
                    ("spikiness", "extrema", "asymmetry")]
    for tri in type_triples:
        combos[f"T_3type_{'_'.join(tri)}"] = list(tri)

    # Large — 4개 이상 대규모 제거 조합
    combos["L_core3"]              = ["buzz", "noise", "momentum"]
    combos["L_core3_cross"]        = ["buzz", "noise", "momentum", "cross"]
    combos["L_core3_multichannel"] = ["buzz", "noise", "momentum", "multichannel"]
    combos["L_basic6"]             = ["momentum", "buzz", "noise", "dead", "wave", "surge"]
    combos["L_ar_cluster"]         = ["autoregressive", "lag", "zscore", "buzz"]
    combos["L_extreme_simplify"]   = ["buzz", "community", "noise", "daily_momentum", "calendar"]
    for keep in channels:
        combos[f"L_keep_{keep}"] = [c for c in channels if c != keep]
    combos["L_drop_all_small"]    = ["calendar", "peak_recency", "asymmetry", "skewkurt", "rsi"]
    combos["L_drop_trend_family"] = ["slope", "band", "growth", "rsi"]

    # Ablation — 가설 검증용 대조 짝
    combos["A_only_small_masks"] = ["calendar", "peak_recency", "asymmetry"]
    combos["A_ch_search_vs_all"] = ["click", "blog", "instagram", "community"]
    combos["A_ch_click_vs_all"]  = ["search", "blog", "instagram", "community"]
    combos["A_noise_minus_dead"] = ["wave", "surge", "dead"]
    combos["A_pivot_noise_only"] = ["dead", "wave", "surge"]

    # MASK_REGISTRY에 없는 마스크 자동 제외 — 결과 동일 조합은 dedupe
    cleaned = {}
    seen_keys = {}
    for name, mask_list in combos.items():
        valid = [m for m in mask_list if m in MASK_REGISTRY]
        if name != "baseline" and mask_list and not valid:
            continue
        key = tuple(sorted(valid))
        if key in seen_keys:
            continue
        seen_keys[key] = name
        cleaned[name] = valid
    return cleaned


MASK_COMBOS = _build_combos()


def validate_coverage():
    # 마스크 등장 횟수 카운터 — 한 마스크라도 빠지지 않는지 검증
    from collections import Counter
    counter = Counter()
    for mask_list in MASK_COMBOS.values():
        for m in mask_list:
            counter[m] += 1
    return counter


_coverage = validate_coverage()
print(f"[config] MASK_REGISTRY={len(MASK_REGISTRY)}, MASK_COMBOS={len(MASK_COMBOS)}, "
      f"min_coverage={min(_coverage.values())}, max_coverage={max(_coverage.values())}")
assert all(v >= 1 for v in _coverage.values()), \
    f"등장 0회 마스크: {[k for k, v in _coverage.items() if v == 0]}"
assert len(FEAT_COLS) == 348, f"expected 348, got {len(FEAT_COLS)}"

# LSTM 전용 설정

LSTM_SAMPLE_STRIDE = 3

LSTM_FEAT_COLS = [
    "scale_search", "scale_click", "scale_blog", "scale_insta", "scale_total",
    "vol_search", "vol_click", "vol_blog", "vol_insta",
    "click_search_ratio", "social_search_ratio", "blog_insta_ratio", "active_channels",
    "month_sin", "month_cos",
    "total_trend", "recent_accel", "peak_recency", "activity_ratio",
]

LSTM_LABEL_COLS = [
    "fw_magnitude_5d", "fw_magnitude_10d", "fw_magnitude_15d",
    "fw_growth_10d", "fw_growth_15d",
    "fw_peak_pos_10d", "fw_peak_pos_15d",
    "fw_spike_10d", "fw_spike_15d",
    "fw_cv_10d", "fw_cv_15d",
    "fw_decline_10d", "fw_decline_15d",
]
