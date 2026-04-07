# 271피처 데이터셋 컬럼 정의
# build_dataset.py의 FEAT_COLS 생성 로직과 동일

SIGNALS = ["search", "click", "blog", "instagram"]
W_FULL = [3, 5, 7, 14, 30]
W_OUT = [7, 14, 30]
W_STAT = [7, 14, 30]


def _sig_cols(p):
    c = []
    # A~K: 기존 유지 (3d/5d 제거, rsi_7d/skew_7d/kurt_7d/spikiness_3d,5d 제거)
    for n in W_OUT:     c.append(f"{p}_volume_{n}d")
    for n in W_OUT:     c.append(f"{p}_level_{n}d")
    for n in W_OUT:     c.append(f"{p}_growth_{n}d")
    c += [f"{p}_accel_short", f"{p}_accel_mid", f"{p}_accel_long"]
    for n in [14, 30]:  c.append(f"{p}_rsi_{n}d")
    for n in [14, 30]:  c.append(f"{p}_skew_{n}d")
    for n in [14, 30]:  c.append(f"{p}_kurt_{n}d")
    for n in W_OUT:     c.append(f"{p}_spikiness_{n}d")
    for n in W_STAT:    c.append(f"{p}_cv_{n}d")
    c.append(f"{p}_cv_change")
    for n in [1, 3, 7, 14, 30]: c.append(f"{p}_lag_{n}d")
    for n in W_STAT:    c.append(f"{p}_std_{n}d")
    for n in W_FULL:    c.append(f"{p}_pos_{n}d")
    c.append(f"days_since_{p}_max_14d")
    # L. Micro-Trend (8)
    c += [f"{p}_slope_3d", f"{p}_slope_7d", f"{p}_slope_14d"]
    c += [f"{p}_norm_slope_7d", f"{p}_slope_change"]
    c += [f"{p}_trend_r2_7d", f"{p}_residual_energy_7d", f"{p}_curvature_14d"]
    # M. Wave Dynamics (6)
    c += [f"{p}_vol_ratio", f"{p}_range_squeeze", f"{p}_damping"]
    c += [f"{p}_zero_crossings_14d", f"{p}_direction_streak", f"{p}_reversal_3d"]
    # N. Regime/Breakout (4)
    c += [f"{p}_band_position", f"{p}_band_width", f"{p}_drawdown_14d", f"{p}_breakout_strength"]
    # O. Daily Momentum (4)
    c += [f"{p}_daily_return_1d", f"{p}_daily_returns_3d_avg"]
    c += [f"{p}_momentum_divergence", f"{p}_daily_accel_1d"]
    return c  # 61개


FEAT_COLS = []
for _p in SIGNALS:
    FEAT_COLS += _sig_cols(_p)

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
]
FEAT_COLS += [
    "blog_surge_intensity", "blog_surge_accel", "blog_surge_lead_days",
    "instagram_surge_intensity", "instagram_surge_accel", "instagram_surge_lead_days",
]
FEAT_COLS += ["month"]

LABEL_COLS = [
    # v1
    "future_intensity", "future_acceleration",
    "signal_agreement", "search_click_convergence",
    # v2
    "buzz_composite", "momentum_score",
    "channel_breadth", "conversion_shift",
    "buzz_rank",
]
META_COLS = ["keyword", "date"]
ALL_COLS = META_COLS + FEAT_COLS + LABEL_COLS

# data_loader 검증 시 피처만 필수, 라벨은 --target에 따라 하나만 사용
REQUIRED_COLS = FEAT_COLS
OPTIONAL_COLS = META_COLS + LABEL_COLS

assert len(FEAT_COLS) == 271, f"expected 271, got {len(FEAT_COLS)}"
