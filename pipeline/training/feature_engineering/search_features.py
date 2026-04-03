"""
검색량·클릭수 기반 피처 계산 모듈

입력 : date, name, search(검색량), click(클릭수) 컬럼을 가진 DataFrame
출력 : 아래 54개 피처 컬럼이 추가된 DataFrame

  [검색량 기반 27개]
    이동평균: search_ma_3d/5d/7d/14d/30d
    성장률: search_growth_3d/5d/7d/14d/30d
    기울기: search_slope_3d/5d/7d/14d/30d
    변동성: search_std_7d/14d/30d
    위치: search_pos_7d/14d/30d
    최고점 경과일: days_since_search_max_7d/14d/30d
    가속도: search_accel_short/mid/long

  [클릭수 기반 27개]
    동일 구조 (click_* 접두사)

  [검색+클릭 결합 6개]
    click_search_ratio, click_lead_7d, click_lead_30d,
    engage_force_7d, engage_force_30d, click_peak_gap

모든 피처는 최근 60일 데이터 기준으로 계산.
윈도우 초기(과거 데이터 부족)는 NaN — NaN 처리는 호출자가 결정.

사용법:
  from feature_engineering.search_features import compute_all_features
  df = compute_all_features(raw_df)
"""

import numpy as np
import pandas as pd

_EPS = 1e-6  # 0 나눗셈 방지


# ── 내부 유틸 ─────────────────────────────────────────────────────────────────

def _growth(s: pd.Series, n: int) -> pd.Series:
    """mean(최근 n일) / mean(이전 n일) − 1"""
    recent = s.rolling(n, min_periods=n).mean()
    prev   = s.shift(n).rolling(n, min_periods=n).mean()
    return recent / (prev + _EPS) - 1


def _slope(s: pd.Series, n: int) -> pd.Series:
    """최근 n일 선형회귀 기울기"""
    return s.rolling(n, min_periods=2).apply(
        lambda x: float(np.polyfit(np.arange(len(x)), x, 1)[0]),
        raw=True,
    )


def _days_since_max(s: pd.Series, n: int) -> pd.Series:
    """최근 n일 내 max 이후 경과일 (오늘이 max면 0)"""
    return s.rolling(n, min_periods=1).apply(
        lambda x: float(len(x) - 1 - int(np.argmax(x))),
        raw=True,
    )


# ── 키워드별 피처 계산 ────────────────────────────────────────────────────────

def _features_per_name(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    s = g["search"]
    c = g["click"]

    # ── 검색량 기반 (27개) ───────────────────────────────────────────────────
    # 이동평균 (5개)
    g["search_ma_3d"]      = s.rolling(3,  min_periods=1).mean()
    g["search_ma_5d"]      = s.rolling(5,  min_periods=1).mean()
    g["search_ma_7d"]      = s.rolling(7,  min_periods=1).mean()
    g["search_ma_14d"]     = s.rolling(14, min_periods=1).mean()
    g["search_ma_30d"]     = s.rolling(30, min_periods=1).mean()

    # 성장률 (5개)
    g["search_growth_3d"]  = _growth(s, 3)
    g["search_growth_5d"]  = _growth(s, 5)
    g["search_growth_7d"]  = _growth(s, 7)
    g["search_growth_14d"] = _growth(s, 14)
    g["search_growth_30d"] = _growth(s, 30)

    # 기울기 (5개)
    g["search_slope_3d"]   = _slope(s, 3)
    g["search_slope_5d"]   = _slope(s, 5)
    g["search_slope_7d"]   = _slope(s, 7)
    g["search_slope_14d"]  = _slope(s, 14)
    g["search_slope_30d"]  = _slope(s, 30)

    # 변동성 (3개)
    g["search_std_7d"]     = s.rolling(7,  min_periods=2).std()
    g["search_std_14d"]    = s.rolling(14, min_periods=2).std()
    g["search_std_30d"]    = s.rolling(30, min_periods=2).std()

    # 위치 (3개)
    g["search_pos_7d"]     = s / (s.rolling(7,  min_periods=1).max() + _EPS)
    g["search_pos_14d"]    = s / (s.rolling(14, min_periods=1).max() + _EPS)
    g["search_pos_30d"]    = s / (s.rolling(30, min_periods=1).max() + _EPS)

    # 최고점 경과일 (3개)
    g["days_since_search_max_7d"]  = _days_since_max(s, 7)
    g["days_since_search_max_14d"] = _days_since_max(s, 14)
    g["days_since_search_max_30d"] = _days_since_max(s, 30)

    # 가속도 (3개)
    g["search_accel_short"] = g["search_growth_3d"] - g["search_growth_7d"]
    g["search_accel_mid"]   = g["search_growth_7d"] - g["search_growth_14d"]
    g["search_accel_long"]  = g["search_growth_14d"] - g["search_growth_30d"]

    # ── 클릭수 기반 (27개) ───────────────────────────────────────────────────
    # 이동평균 (5개)
    g["click_ma_3d"]       = c.rolling(3,  min_periods=1).mean()
    g["click_ma_5d"]       = c.rolling(5,  min_periods=1).mean()
    g["click_ma_7d"]       = c.rolling(7,  min_periods=1).mean()
    g["click_ma_14d"]      = c.rolling(14, min_periods=1).mean()
    g["click_ma_30d"]      = c.rolling(30, min_periods=1).mean()

    # 성장률 (5개)
    g["click_growth_3d"]   = _growth(c, 3)
    g["click_growth_5d"]   = _growth(c, 5)
    g["click_growth_7d"]   = _growth(c, 7)
    g["click_growth_14d"]  = _growth(c, 14)
    g["click_growth_30d"]  = _growth(c, 30)

    # 기울기 (5개)
    g["click_slope_3d"]    = _slope(c, 3)
    g["click_slope_5d"]    = _slope(c, 5)
    g["click_slope_7d"]    = _slope(c, 7)
    g["click_slope_14d"]   = _slope(c, 14)
    g["click_slope_30d"]   = _slope(c, 30)

    # 변동성 (3개)
    g["click_std_7d"]      = c.rolling(7,  min_periods=2).std()
    g["click_std_14d"]     = c.rolling(14, min_periods=2).std()
    g["click_std_30d"]     = c.rolling(30, min_periods=2).std()

    # 위치 (3개)
    g["click_pos_7d"]      = c / (c.rolling(7,  min_periods=1).max() + _EPS)
    g["click_pos_14d"]     = c / (c.rolling(14, min_periods=1).max() + _EPS)
    g["click_pos_30d"]     = c / (c.rolling(30, min_periods=1).max() + _EPS)

    # 최고점 경과일 (3개)
    g["days_since_click_max_7d"]  = _days_since_max(c, 7)
    g["days_since_click_max_14d"] = _days_since_max(c, 14)
    g["days_since_click_max_30d"] = _days_since_max(c, 30)

    # 가속도 (3개)
    g["click_accel_short"]  = g["click_growth_3d"] - g["click_growth_7d"]
    g["click_accel_mid"]    = g["click_growth_7d"] - g["click_growth_14d"]
    g["click_accel_long"]   = g["click_growth_14d"] - g["click_growth_30d"]

    # ── 검색+클릭 결합 (6개) ─────────────────────────────────────────────────
    g["click_search_ratio"]   = g["click_ma_7d"] / (g["search_ma_7d"] + _EPS)
    g["click_lead_7d"]        = g["click_growth_7d"] - g["search_growth_7d"]
    g["click_lead_30d"]       = g["click_growth_30d"] - g["search_growth_30d"]
    g["engage_force_7d"]      = g["click_ma_7d"] * g["search_slope_7d"]
    g["engage_force_30d"]     = g["click_ma_30d"] * g["search_slope_30d"]
    g["click_peak_gap"]       = g["click_pos_14d"] - g["search_pos_14d"]

    return g


# ── 공개 API ──────────────────────────────────────────────────────────────────

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    검색량·클릭수 기반 피처 60개를 계산해 반환한다.

    Parameters
    ----------
    df : DataFrame
        최소 [date, name, search, click] 컬럼 필요.
        search·click은 일별 수치(float or int).

    Returns
    -------
    DataFrame
        키워드별 날짜 오름차순 정렬, 60개 피처 컬럼 추가
        (검색 27개 + 클릭 27개 + 결합 6개).
    """
    required = {"date", "name", "search", "click"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[search_features] 필수 컬럼 없음: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["name", "date"]).reset_index(drop=True)
    df = df.groupby("name", group_keys=False).apply(_features_per_name)
    return df.reset_index(drop=True)
