"""
검색량·클릭수 기반 피처 계산 모듈

입력 : date, name, search(검색량), click(클릭수) 컬럼을 가진 DataFrame
출력 : 아래 24개 피처 컬럼이 추가된 DataFrame

  [검색량 기반 10개]
    search_ma_7d, search_growth_3d/7d/14d,
    search_slope_7d/14d, search_acceleration,
    search_std_7d, search_pos_14d, days_since_max_14d

  [클릭수 기반 10개]
    click_ma_7d, click_growth_3d/7d/14d,
    click_slope_7d/14d, click_acceleration,
    click_std_7d, click_pos_14d, days_since_click_max_14d

  [검색+클릭 결합 4개]
    click_search_ratio, click_lead_signal_7d,
    engagement_force, click_peak_gap

모든 피처는 최근 60일 데이터 기준으로 계산 (최대 룩백 28일).
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

    # ── 검색량 기반 ──────────────────────────────────────────────────────────
    g["search_ma_7d"]      = s.rolling(7,  min_periods=1).mean()
    g["search_growth_3d"]  = _growth(s, 3)
    g["search_growth_7d"]  = _growth(s, 7)
    g["search_growth_14d"] = _growth(s, 14)
    g["search_slope_7d"]   = _slope(s, 7)
    g["search_slope_14d"]  = _slope(s, 14)
    g["search_acceleration"] = g["search_growth_7d"] - g["search_growth_14d"]
    g["search_std_7d"]     = s.rolling(7,  min_periods=2).std()
    g["search_pos_14d"]    = s / (s.rolling(14, min_periods=1).max() + _EPS)
    g["days_since_max_14d"] = _days_since_max(s, 14)

    # ── 클릭수 기반 ──────────────────────────────────────────────────────────
    g["click_ma_7d"]       = c.rolling(7,  min_periods=1).mean()
    g["click_growth_3d"]   = _growth(c, 3)
    g["click_growth_7d"]   = _growth(c, 7)
    g["click_growth_14d"]  = _growth(c, 14)
    g["click_slope_7d"]    = _slope(c, 7)
    g["click_slope_14d"]   = _slope(c, 14)
    g["click_acceleration"] = g["click_growth_7d"] - g["click_growth_14d"]
    g["click_std_7d"]      = c.rolling(7,  min_periods=2).std()
    g["click_pos_14d"]     = c / (c.rolling(14, min_periods=1).max() + _EPS)
    g["days_since_click_max_14d"] = _days_since_max(c, 14)

    # ── 검색+클릭 결합 ───────────────────────────────────────────────────────
    g["click_search_ratio"]   = g["click_ma_7d"] / (g["search_ma_7d"] + _EPS)
    g["click_lead_signal_7d"] = g["click_growth_7d"] - g["search_growth_7d"]
    g["engagement_force"]     = g["click_ma_7d"] * g["search_slope_7d"]
    g["click_peak_gap"]       = g["click_pos_14d"] - g["search_pos_14d"]

    return g


# ── 공개 API ──────────────────────────────────────────────────────────────────

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    검색량·클릭수 기반 피처 24개를 계산해 반환한다.

    Parameters
    ----------
    df : DataFrame
        최소 [date, name, search, click] 컬럼 필요.
        search·click은 일별 수치(float or int).

    Returns
    -------
    DataFrame
        키워드별 날짜 오름차순 정렬, 24개 피처 컬럼 추가.
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
