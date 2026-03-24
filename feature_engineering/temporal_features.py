"""
날짜/시간 기반 피처 및 경쟁 트렌드 피처 계산 모듈

계산 피처:
  D1. day_of_week             — 요일 (0=월 ~ 6=일)
  D2. month                   — 월 (1~12)
  D3. is_holiday              — 한국 공휴일 여부 (0 or 1)
  D4. competing_trends_count  — 같은 날 virality_score > 1.5인 다른 키워드 수

사용법:
  from feature_engineering.temporal_features import compute_temporal_features
  df = compute_temporal_features(df)   # date, keyword, (virality_score) 포함 DataFrame
"""

import warnings

import holidays
import pandas as pd

# D4 판정 기준: 이 값을 초과하는 키워드를 "바이럴 중"으로 간주
_VIRAL_THRESHOLD = 1.5


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """D1, D2: date 컬럼에서 요일·월 추출"""
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=월요일, 6=일요일
    df["month"] = df["date"].dt.month             # 1~12
    return df


def _add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """D3: 한국 공휴일 여부 (설·추석 연휴, 크리스마스, 어린이날 등 포함)"""
    years = df["date"].dt.year.unique().tolist()
    kr_holidays = holidays.KR(years=years)
    df["is_holiday"] = df["date"].isin(kr_holidays).astype(int)
    return df


def _add_competing_trends(df: pd.DataFrame) -> pd.DataFrame:
    """
    D4: 같은 날 virality_score > 1.5 인 다른 키워드 수.

    virality_score 컬럼이 없으면 경고를 출력하고 0으로 채운다.
    """
    if "virality_score" not in df.columns:
        warnings.warn(
            "[temporal_features] virality_score 컬럼이 없어 "
            "competing_trends_count를 0으로 채웁니다.",
            UserWarning,
            stacklevel=3,
        )
        df["competing_trends_count"] = 0
        return df

    # 날짜별로 virality_score > 1.5 인 키워드 수 집계
    viral_per_date = (
        df[df["virality_score"] > _VIRAL_THRESHOLD]
        .groupby("date")["keyword"]
        .count()
        .rename("_viral_count")
    )
    df = df.join(viral_per_date, on="date")

    # 자기 자신이 threshold를 넘을 경우 본인 제외 (경쟁자 = 타인만)
    is_self_viral = (df["virality_score"] > _VIRAL_THRESHOLD).astype(int)
    df["competing_trends_count"] = (
        df["_viral_count"].fillna(0) - is_self_viral
    ).clip(lower=0).astype(int)

    df = df.drop(columns=["_viral_count"])
    return df


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    날짜/경쟁 피처를 계산해 반환한다.

    Parameters
    ----------
    df : DataFrame
        최소한 [date, keyword] 컬럼을 포함해야 함.
        D4(competing_trends_count)를 계산하려면 virality_score 컬럼도 필요.

    Returns
    -------
    DataFrame
        입력 df에 D1~D4 피처 컬럼이 추가된 DataFrame.
    """
    required = {"date", "keyword"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[temporal_features] 필수 컬럼 없음: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = _add_date_features(df)
    df = _add_holiday_flag(df)
    df = _add_competing_trends(df)

    return df
