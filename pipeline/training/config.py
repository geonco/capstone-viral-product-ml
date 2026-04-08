# 348피처 + 15라벨 데이터셋 컬럼 정의
# build_dataset.py의 FEAT_COLS/LABEL_COLS 생성 로직과 동일

SIGNALS = ["search", "click", "blog", "instagram"]
W_FULL = [3, 5, 7, 14, 30]
W_STAT = [7, 14, 30]


W_SHORT = [3, 7, 14, 30]

def _sig_cols(p):
    c = []
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
    # L. Micro-Trend (8)
    c += [f"{p}_slope_3d", f"{p}_slope_7d", f"{p}_slope_14d"]
    c += [f"{p}_norm_slope_7d", f"{p}_slope_change"]
    c += [f"{p}_trend_r2_7d", f"{p}_residual_energy_7d", f"{p}_curvature_14d"]
    # M. Wave Dynamics (6)
    c += [f"{p}_vol_ratio", f"{p}_range_squeeze", f"{p}_damping"]
    c += [f"{p}_zero_crossings_14d", f"{p}_direction_streak", f"{p}_reversal_3d"]
    # N. Regime/Breakout (5)
    c += [f"{p}_band_position", f"{p}_band_width", f"{p}_drawdown_14d", f"{p}_breakout_strength"]
    c.append(f"{p}_band_approach")
    # O. Daily Momentum (4)
    c += [f"{p}_daily_return_1d", f"{p}_daily_returns_3d_avg"]
    c += [f"{p}_momentum_divergence", f"{p}_daily_accel_1d"]
    # P. Daily Change (7)
    for d in range(1, 8):
        c.append(f"{p}_change_{d}d_ago")
    return c  # 73개


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
    "blog_search_lead_magnitude", "insta_search_lead_magnitude", "click_search_lead_magnitude",
    "weighted_slope_strength", "activation_spread",
]
FEAT_COLS += [
    "blog_surge_intensity_3d", "blog_surge_intensity_7d",
    "blog_surge_accel_3d", "blog_surge_accel_7d", "blog_surge_lead_days",
    "instagram_surge_intensity_3d", "instagram_surge_intensity_7d",
    "instagram_surge_accel_3d", "instagram_surge_accel_7d", "instagram_surge_lead_days",
]
_W_MULTI = [1, 3, 7, 14, 30]
for _n in _W_MULTI:
    FEAT_COLS += [f"past_conversion_rate_{_n}d", f"past_channel_active_{_n}d"]
    if _n >= 3:
        FEAT_COLS.append(f"multi_accel_{_n}d")
    FEAT_COLS.append(f"past_buzz_zscore_{_n}d")
FEAT_COLS += ["conv_trend"]
FEAT_COLS += ["month"]

LABEL_COLS = [
    "intensity_5d", "intensity_10d", "intensity_15d",
    "buzz_composite_5d", "buzz_composite_10d", "buzz_composite_15d",
    "growth_5d", "growth_10d", "growth_15d",
    "sustainability_5d", "sustainability_10d", "sustainability_15d",
    "crash_5d", "crash_10d", "crash_15d",
]
META_COLS = ["keyword", "date"]
ALL_COLS = META_COLS + FEAT_COLS + LABEL_COLS

REQUIRED_COLS = FEAT_COLS
OPTIONAL_COLS = META_COLS + LABEL_COLS

assert len(FEAT_COLS) == 348, f"expected 348, got {len(FEAT_COLS)}"
