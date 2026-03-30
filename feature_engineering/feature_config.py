# ── 메타 컬럼 (식별자, 모델 입력에 사용하지 않음) ───────────────────────────────
META_COLS: list[str] = [
    "keyword",
    "date",
]

# ── 원시 입력 컬럼 (피처 계산의 재료, 모델 입력 아님) ──────────────────────────
RAW_COLS: list[str] = [
    "search",   # 일별 검색량
    "click",    # 일별 클릭수
]

# ── 피처 컬럼 (모델 입력, canonical 순서) ────────────────────────────────────────
# 새 feature를 추가할 때는 이 리스트에만 추가하면 됩니다.
FEATURE_COLS: list[str] = [
    # 검색량 기반 (10)
    "search_ma_7d",
    "search_growth_3d",
    "search_growth_7d",
    "search_growth_14d",
    "search_slope_7d",
    "search_slope_14d",
    "search_acceleration",
    "search_std_7d",
    "search_pos_14d",
    "days_since_max_14d",
    # 클릭수 기반 (10)
    "click_ma_7d",
    "click_growth_3d",
    "click_growth_7d",
    "click_growth_14d",
    "click_slope_7d",
    "click_slope_14d",
    "click_acceleration",
    "click_std_7d",
    "click_pos_14d",
    "days_since_click_max_14d",
    # 검색+클릭 결합 (4)
    "click_search_ratio",
    "click_lead_signal_7d",
    "engagement_force",
    "click_peak_gap",
    # 언급량 기반 (9)
    "mention_ma_3d",
    "mention_ma_7d",
    "mention_growth_3d",
    "mention_growth_7d",
    "mention_growth_14d",
    "mention_acceleration",
    "mention_std_7d",
    "mention_pos_14d",
    "days_since_mention_max_14d",
    # 검색-언급 결합 (4)
    "mention_search_ratio",
    "lead_signal_7d",
    "viral_force",
    "peak_gap",
    # 클릭 비율 (14)
    "male_click_ratio",
    "female_click_ratio",
    "gender_click_skew",
    "gender_click_shift_7d",
    "age10_click_ratio",
    "age20_click_ratio",
    "age30_click_ratio",
    "age40_click_ratio",
    "age50p_click_ratio",
    "young_click_ratio",
    "mid_click_ratio",
    "core_age_click_ratio",
    "age_click_entropy",
    "age_click_shift_7d",
]

# ── 타겟 컬럼 (모델 출력) ──────────────────────────────────────────────────────
TARGET_COLS: list[str] = [
    "virality_score",
    "peak_time",
]

# ── 파생 그룹 (위 리스트에서 자동 생성) ────────────────────────────────────────
ALL_COLS: list[str] = META_COLS + FEATURE_COLS + TARGET_COLS  # 총 55개

# 없으면 ValueError (학습 불가)
REQUIRED_COLS: list[str] = FEATURE_COLS + TARGET_COLS

# 없으면 경고만 (학습 가능)
OPTIONAL_COLS: list[str] = ["keyword", "date"]
