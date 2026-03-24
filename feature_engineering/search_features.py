"""
검색 트렌드 기반 피처 계산 모듈

입력: date, keyword, ratio 컬럼을 가진 DataFrame
출력: 아래 피처 컬럼이 추가된 DataFrame

계산 피처:
  A1. search_velocity_3d / 7d / 14d  — 검색 변화율
  A2. search_acceleration             — 검색 가속도 (velocity_7d 기반)
  A3. ma_ratio_3_14 / 7_30           — 이동평균 비율
  A4. search_volatility_3d / 7d / 14d — 검색 변동성
  A5. trend_slope_30d                 — 30일 선형 추세 기울기

사용법:
  from feature_engineering.search_features import compute_search_features
  df = compute_search_features(raw_df)   # raw_df: [date, keyword, ratio, ...]
"""

import numpy as np
import pandas as pd


def _slope(values: np.ndarray) -> float:
    """values 배열에 선형회귀를 적합하고 기울기를 반환한다."""
    n = len(values)
    if n < 2:
        return np.nan
    x = np.arange(n, dtype=float)
    return float(np.polyfit(x, values, 1)[0])


def _features_per_keyword(g: pd.DataFrame) -> pd.DataFrame:
    """단일 키워드 그룹에 대해 모든 검색 피처를 계산한다."""
    r = g["ratio"]

    # ── A1. search_velocity ───────────────────────────────────────────────────
    # (trend_today - trend_N_days_ago) / (trend_N_days_ago + 1)
    # shift(N): N행 앞의 값 = N일 전 (날짜순 정렬 기준)
    g = g.copy()
    g["search_velocity_3d"]  = (r - r.shift(3))  / (r.shift(3)  + 1)
    g["search_velocity_7d"]  = (r - r.shift(7))  / (r.shift(7)  + 1)
    g["search_velocity_14d"] = (r - r.shift(14)) / (r.shift(14) + 1)

    # ── A2. search_acceleration ───────────────────────────────────────────────
    # velocity_7d 자체의 1일 변화량
    # velocity_7d 계산 이후에 shift 적용
    v7 = g["search_velocity_7d"]
    g["search_acceleration"] = v7 - v7.shift(1)

    # ── A3. ma_ratio ──────────────────────────────────────────────────────────
    # 이동평균 비율: 단기 MA / (장기 MA + 1)
    ma3  = r.rolling(3,  min_periods=1).mean()
    ma7  = r.rolling(7,  min_periods=1).mean()
    ma14 = r.rolling(14, min_periods=1).mean()
    ma30 = r.rolling(30, min_periods=1).mean()

    g["ma_ratio_3_14"] = ma3 / (ma14 + 1)
    g["ma_ratio_7_30"] = ma7 / (ma30 + 1)

    # ── A4. search_volatility ─────────────────────────────────────────────────
    # 최근 N일 일별 트렌드의 표준편차
    g["search_volatility_3d"]  = r.rolling(3,  min_periods=2).std()
    g["search_volatility_7d"]  = r.rolling(7,  min_periods=2).std()
    g["search_volatility_14d"] = r.rolling(14, min_periods=2).std()

    # ── A5. trend_slope_30d ───────────────────────────────────────────────────
    # 최근 30일 (일자, 트렌드값)에 선형회귀 적합 → 기울기
    g["trend_slope_30d"] = r.rolling(30, min_periods=2).apply(_slope, raw=True)

    return g


def compute_search_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    검색 트렌드 피처를 계산해 반환한다.

    Parameters
    ----------
    df : DataFrame
        최소한 [date, keyword, ratio] 컬럼을 포함해야 함.
        date 컬럼은 날짜 파싱 가능한 형식이어야 함.

    Returns
    -------
    DataFrame
        입력 df에 피처 컬럼이 추가된 DataFrame.
        키워드별 날짜 오름차순 정렬 상태로 반환됨.

    Notes
    -----
    - 윈도우 초기(충분한 과거 데이터 없음)는 NaN으로 채워짐.
      예: velocity_14d는 최소 14일치 데이터가 있어야 유효.
    - NaN 처리(dropna / fillna)는 호출자가 결정한다.
    """
    required = {"date", "keyword", "ratio"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[search_features] 필수 컬럼 없음: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["keyword", "date"]).reset_index(drop=True)

    df = df.groupby("keyword", group_keys=False).apply(_features_per_keyword)

    return df.reset_index(drop=True)
