# ── 메타 컬럼 (식별자, 모델 입력에 사용하지 않음) ───────────────────────────────
META_COLS: list[str] = [
    "date",
    "keyword",
]

# ── 피처 컬럼 (모델 입력, canonical 순서) ────────────────────────────────────────
# 새 feature를 추가할 때는 이 리스트에만 추가하면 됩니다.
FEATURE_COLS: list[str] = [
    "ratio",
    "search_velocity_3d",
    "search_velocity_7d",
    "search_velocity_14d",
    "search_acceleration",
    "ma_ratio_3_14",
    "ma_ratio_7_30",
    "search_volatility_3d",
    "search_volatility_7d",
    "search_volatility_14d",
    "trend_slope_30d",
    "day_of_week",
    "month",
    "is_holiday",
    "competing_trends_count",
    "stratum_code",
    "appearance",
    "best_rank",
    "rank_std",
]

# ── 타겟 컬럼 (모델 출력) ──────────────────────────────────────────────────────
TARGET_COLS: list[str] = [
    "virality_score",
    "peak_timing",
    "is_viral",
]

# ── 파생 그룹 (위 리스트에서 자동 생성) ────────────────────────────────────────
ALL_COLS: list[str] = META_COLS + FEATURE_COLS + TARGET_COLS

# 없으면 ValueError (학습 불가)
REQUIRED_COLS: list[str] = FEATURE_COLS + ["virality_score", "peak_timing"]

# 없으면 경고만 (학습 가능)
OPTIONAL_COLS: list[str] = ["date", "keyword", "is_viral"]
