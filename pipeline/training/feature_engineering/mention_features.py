"""
언급량(SNS) 기반 피처 계산 모듈

입력 : date, name, search, mention 컬럼을 가진 DataFrame
출력 : 아래 33개 피처 컬럼이 추가된 DataFrame

  [언급량 기반 27개]
    이동평균: mention_ma_3d/5d/7d/14d/30d
    성장률: mention_growth_3d/5d/7d/14d/30d
    기울기: mention_slope_3d/5d/7d/14d/30d
    변동성: mention_std_7d/14d/30d
    위치: mention_pos_7d/14d/30d
    최고점 경과일: days_since_mention_max_7d/14d/30d
    가속도: mention_accel_short/mid/long

  [검색+언급 결합 6개]
    mention_search_ratio, mention_lead_7d, mention_lead_30d,
    viral_force_7d, viral_force_30d, mention_peak_gap

모든 피처는 최근 60일 데이터 기준으로 계산.
윈도우 초기(과거 데이터 부족)는 NaN — NaN 처리는 호출자가 결정.

사용법:
  from feature_engineering.mention_features import compute_mention_features
  df = compute_mention_features(raw_df)
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
    m = g["mention"]
    s = g["search"]

    # ── 언급량 기반 (27개) ───────────────────────────────────────────────────
    # 이동평균 (5개)
    g["mention_ma_3d"]      = m.rolling(3,  min_periods=1).mean()
    g["mention_ma_5d"]      = m.rolling(5,  min_periods=1).mean()
    g["mention_ma_7d"]      = m.rolling(7,  min_periods=1).mean()
    g["mention_ma_14d"]     = m.rolling(14, min_periods=1).mean()
    g["mention_ma_30d"]     = m.rolling(30, min_periods=1).mean()

    # 성장률 (5개)
    g["mention_growth_3d"]  = _growth(m, 3)
    g["mention_growth_5d"]  = _growth(m, 5)
    g["mention_growth_7d"]  = _growth(m, 7)
    g["mention_growth_14d"] = _growth(m, 14)
    g["mention_growth_30d"] = _growth(m, 30)

    # 기울기 (5개)
    g["mention_slope_3d"]   = _slope(m, 3)
    g["mention_slope_5d"]   = _slope(m, 5)
    g["mention_slope_7d"]   = _slope(m, 7)
    g["mention_slope_14d"]  = _slope(m, 14)
    g["mention_slope_30d"]  = _slope(m, 30)

    # 변동성 (3개)
    g["mention_std_7d"]     = m.rolling(7,  min_periods=2).std()
    g["mention_std_14d"]    = m.rolling(14, min_periods=2).std()
    g["mention_std_30d"]    = m.rolling(30, min_periods=2).std()

    # 위치 (3개)
    g["mention_pos_7d"]     = m / (m.rolling(7,  min_periods=1).max() + _EPS)
    g["mention_pos_14d"]    = m / (m.rolling(14, min_periods=1).max() + _EPS)
    g["mention_pos_30d"]    = m / (m.rolling(30, min_periods=1).max() + _EPS)

    # 최고점 경과일 (3개)
    g["days_since_mention_max_7d"]  = _days_since_max(m, 7)
    g["days_since_mention_max_14d"] = _days_since_max(m, 14)
    g["days_since_mention_max_30d"] = _days_since_max(m, 30)

    # 가속도 (3개)
    g["mention_accel_short"] = g["mention_growth_3d"] - g["mention_growth_7d"]
    g["mention_accel_mid"]   = g["mention_growth_7d"] - g["mention_growth_14d"]
    g["mention_accel_long"]  = g["mention_growth_14d"] - g["mention_growth_30d"]

    # ── 검색+언급 결합 (6개) ─────────────────────────────────────────────────
    g["mention_search_ratio"] = g["mention_ma_7d"] / (s.rolling(7, min_periods=1).mean() + _EPS)
    g["mention_lead_7d"]      = g["mention_growth_7d"] - _growth(s, 7)
    g["mention_lead_30d"]     = g["mention_growth_30d"] - _growth(s, 30)
    g["viral_force_7d"]       = g["mention_ma_7d"] * _slope(s, 7)
    g["viral_force_30d"]      = g["mention_ma_30d"] * _slope(s, 30)
    g["mention_peak_gap"]     = (m / (m.rolling(14, min_periods=1).max() + _EPS)) - \
                                (s / (s.rolling(14, min_periods=1).max() + _EPS))

    return g


# ── 공개 API ──────────────────────────────────────────────────────────────────

def compute_mention_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    언급량 기반 피처 33개를 계산해 반환한다.

    Parameters
    ----------
    df : DataFrame
        최소 [date, name, search, mention] 컬럼 필요.
        search·mention은 일별 수치(float or int).

    Returns
    -------
    DataFrame
        키워드별 날짜 오름차순 정렬, 33개 피처 컬럼 추가.
    """
    required = {"date", "name", "search", "mention"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[mention_features] 필수 컬럼 없음: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["name", "date"]).reset_index(drop=True)
    df = df.groupby("name", group_keys=False).apply(_features_per_name)
    return df.reset_index(drop=True)
