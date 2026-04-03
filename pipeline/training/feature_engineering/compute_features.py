"""
통합 피처 계산 모듈

검색, 클릭, 언급, 인구통계 기반의 모든 피처(107개)를 한 번에 계산한다.

입력: raw 데이터 (keyword, date, search, click, mention, demographic 컬럼)
출력: 107개 피처가 추가된 DataFrame (feature_config.FEATURE_COLS 순서)

사용법:
  from feature_engineering.compute_features import compute_all_features_unified
  df_features = compute_all_features_unified(df_raw)
"""

import pandas as pd

from feature_engineering.feature_config import FEATURE_COLS
from feature_engineering.search_features import compute_all_features as compute_search_click
from feature_engineering.mention_features import compute_mention_features
from feature_engineering.demographic_features import compute_demographic_features


def compute_all_features_unified(df: pd.DataFrame) -> pd.DataFrame:
    """
    raw 데이터에서 모든 107개 피처를 계산한다.

    Parameters
    ----------
    df : DataFrame
        필수 컬럼:
          - 식별자: keyword, date
          - 신호원: search, click, mention
          - 인구통계: gender_m_pct, gender_f_pct, age_10_pct, age_20_pct, 
                      age_30_pct, age_40_pct, age_50_pct, age_60_pct
        date는 datetime으로 캐스트됨.

    Returns
    -------
    DataFrame
        모든 메타/피처/타겟 컬럼이 포함된 데이터프레임.
        피처는 feature_config.FEATURE_COLS 순서대로 배열됨.

    Raises
    ------
    ValueError
        필수 컬럼이 누락된 경우.
    """
    # 1. 입력 검증
    required = {"keyword", "date", "search", "click", "mention"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[compute_all_features_unified] 필수 컬럼 누락: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 2. 검색+클릭 피처 계산 (60개: search 27 + click 27 + combined 6)
    df = compute_search_click(df)

    # 3. 언급 피처 계산 (33개: mention 27 + combined 6)
    df = compute_mention_features(df)

    # 4. 인구통계 피처 계산 (14개)
    df = compute_demographic_features(df)

    # 5. 피처 순서 정렬 (feature_config.FEATURE_COLS 순서 유지)
    feature_cols_present = [c for c in FEATURE_COLS if c in df.columns]
    missing_features = set(FEATURE_COLS) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"[compute_all_features_unified] 계산 후 누락된 피처: {sorted(missing_features)}"
        )

    # 모든 컬럼을 메타 + 피처 + 나머지 순서로 배열
    meta_cols = ["keyword", "date"]
    raw_cols = ["search", "click", "mention"]
    existing_cols = set(df.columns)
    
    # 메타, raw, 피처 순서 유지
    ordered_cols = [c for c in meta_cols if c in existing_cols]
    ordered_cols += [c for c in raw_cols if c in existing_cols]
    ordered_cols += feature_cols_present
    
    # 나머지 컬럼(타겟 포함)
    extra_cols = [c for c in df.columns if c not in set(ordered_cols)]
    ordered_cols += extra_cols

    return df[ordered_cols]
