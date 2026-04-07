# 216피처 데이터셋 컬럼 정의
# build_dataset_20260403.py의 FEAT_COLS 생성 로직과 동일

SIGNALS = ["search", "click", "blog", "instagram"]
W_FULL = [3, 5, 7, 14, 30]
W_STAT = [7, 14, 30]


def _sig_cols(p):
    c = []
    for n in W_FULL: c.append(f"{p}_volume_{n}d")
    for n in W_FULL: c.append(f"{p}_level_{n}d")
    for n in W_FULL: c.append(f"{p}_growth_{n}d")
    c += [f"{p}_accel_short", f"{p}_accel_mid", f"{p}_accel_long"]
    for n in W_STAT: c.append(f"{p}_rsi_{n}d")
    for n in W_STAT: c.append(f"{p}_skew_{n}d")
    for n in W_STAT: c.append(f"{p}_kurt_{n}d")
    for n in W_FULL: c.append(f"{p}_spikiness_{n}d")
    for n in W_STAT: c.append(f"{p}_cv_{n}d")
    c.append(f"{p}_cv_change")
    for n in [1, 3, 7, 14, 30]: c.append(f"{p}_lag_{n}d")
    for n in W_STAT: c.append(f"{p}_std_{n}d")
    for n in W_FULL: c.append(f"{p}_pos_{n}d")
    c.append(f"days_since_{p}_max_14d")
    return c  # 50개


FEAT_COLS = []
for _p in SIGNALS:
    FEAT_COLS += _sig_cols(_p)

FEAT_COLS += [
    "click_search_ratio", "click_lead_7d", "click_lead_14d", "click_search_pos_gap",
    "blog_lead_7d", "blog_search_ratio",
    "instagram_lead_7d", "instagram_search_ratio",
]
FEAT_COLS += [
    "blog_surge_7d", "blog_surge_days_ago", "blog_surge_before_search",
    "instagram_surge_7d", "instagram_surge_days_ago", "instagram_surge_before_search",
]
FEAT_COLS += ["day_of_week", "month"]

LABEL_COLS = [
    "future_intensity", "future_acceleration",
    "signal_agreement", "trajectory_class", "search_click_convergence",
]
META_COLS = ["keyword", "date"]
ALL_COLS = META_COLS + FEAT_COLS + LABEL_COLS

# data_loader 검증 시 피처만 필수, 라벨은 --target에 따라 하나만 사용
REQUIRED_COLS = FEAT_COLS
OPTIONAL_COLS = META_COLS + LABEL_COLS

assert len(FEAT_COLS) == 216, f"expected 216, got {len(FEAT_COLS)}"
