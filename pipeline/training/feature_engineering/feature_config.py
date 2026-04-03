# ── 메타 컬럼 (식별자, 모델 입력에 사용하지 않음) ───────────────────────────────
META_COLS: list[str] = [
    "keyword",
    "date",
]

# ── 원시 입력 컬럼 (피처 계산의 재료, 모델 입력 아님) ──────────────────────────
RAW_COLS: list[str] = [
    "search",   # 일별 검색량
    "click",    # 일별 클릭수
    "mention",  # 일별 언급량 (SNS)
]

# ── 피처 컬럼 (모델 입력, canonical 순서) ────────────────────────────────────────
# v2 피처 테이블 (107개): 4개 신호원(검색/클릭/언급/인구통계) × 다양한 time window
FEATURE_COLS: list[str] = [
    # ── 검색량 기반 (27개) ──────────────────────────────────────────────────
    # 이동평균 (5개)
    "search_ma_3d",
    "search_ma_5d",
    "search_ma_7d",
    "search_ma_14d",
    "search_ma_30d",
    # 성장률 (5개)
    "search_growth_3d",
    "search_growth_5d",
    "search_growth_7d",
    "search_growth_14d",
    "search_growth_30d",
    # 기울기 (5개)
    "search_slope_3d",
    "search_slope_5d",
    "search_slope_7d",
    "search_slope_14d",
    "search_slope_30d",
    # 변동성 (3개)
    "search_std_7d",
    "search_std_14d",
    "search_std_30d",
    # 위치 (3개)
    "search_pos_7d",
    "search_pos_14d",
    "search_pos_30d",
    # 최고점 경과일 (3개)
    "days_since_search_max_7d",
    "days_since_search_max_14d",
    "days_since_search_max_30d",
    # 가속도 (3개)
    "search_accel_short",
    "search_accel_mid",
    "search_accel_long",
    
    # ── 클릭수 기반 (27개) ──────────────────────────────────────────────────
    # 이동평균 (5개)
    "click_ma_3d",
    "click_ma_5d",
    "click_ma_7d",
    "click_ma_14d",
    "click_ma_30d",
    # 성장률 (5개)
    "click_growth_3d",
    "click_growth_5d",
    "click_growth_7d",
    "click_growth_14d",
    "click_growth_30d",
    # 기울기 (5개)
    "click_slope_3d",
    "click_slope_5d",
    "click_slope_7d",
    "click_slope_14d",
    "click_slope_30d",
    # 변동성 (3개)
    "click_std_7d",
    "click_std_14d",
    "click_std_30d",
    # 위치 (3개)
    "click_pos_7d",
    "click_pos_14d",
    "click_pos_30d",
    # 최고점 경과일 (3개)
    "days_since_click_max_7d",
    "days_since_click_max_14d",
    "days_since_click_max_30d",
    # 가속도 (3개)
    "click_accel_short",
    "click_accel_mid",
    "click_accel_long",
    
    # ── 검색+클릭 결합 (6개) ────────────────────────────────────────────────
    "click_search_ratio",
    "click_lead_7d",
    "click_lead_30d",
    "engage_force_7d",
    "engage_force_30d",
    "click_peak_gap",
    
    # ── 언급량 기반 (27개) ──────────────────────────────────────────────────
    # 이동평균 (5개)
    "mention_ma_3d",
    "mention_ma_5d",
    "mention_ma_7d",
    "mention_ma_14d",
    "mention_ma_30d",
    # 성장률 (5개)
    "mention_growth_3d",
    "mention_growth_5d",
    "mention_growth_7d",
    "mention_growth_14d",
    "mention_growth_30d",
    # 기울기 (5개)
    "mention_slope_3d",
    "mention_slope_5d",
    "mention_slope_7d",
    "mention_slope_14d",
    "mention_slope_30d",
    # 변동성 (3개)
    "mention_std_7d",
    "mention_std_14d",
    "mention_std_30d",
    # 위치 (3개)
    "mention_pos_7d",
    "mention_pos_14d",
    "mention_pos_30d",
    # 최고점 경과일 (3개)
    "days_since_mention_max_7d",
    "days_since_mention_max_14d",
    "days_since_mention_max_30d",
    # 가속도 (3개)
    "mention_accel_short",
    "mention_accel_mid",
    "mention_accel_long",
    
    # ── 검색+언급 결합 (6개) ────────────────────────────────────────────────
    "mention_search_ratio",
    "mention_lead_7d",
    "mention_lead_30d",
    "viral_force_7d",
    "viral_force_30d",
    "mention_peak_gap",
    
    # ── 인구통계 (14개) ─────────────────────────────────────────────────────
    # 성별 (4개)
    "male_click_ratio",
    "female_click_ratio",
    "gender_click_skew",
    "gender_click_shift_7d",
    # 연령대 (10개)
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
    "log_virality_score",
    "peak_time",
]

# ── 파생 그룹 (위 리스트에서 자동 생성) ────────────────────────────────────────
ALL_COLS: list[str] = META_COLS + RAW_COLS + FEATURE_COLS + TARGET_COLS  # 총 127개

# 없으면 ValueError (학습 불가)
REQUIRED_COLS: list[str] = FEATURE_COLS + TARGET_COLS

# 없으면 경고만 (학습 가능한 경우가 있을 수 있음)
OPTIONAL_COLS: list[str] = ["keyword", "date"]
