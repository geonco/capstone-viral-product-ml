"""
인구통계(Demographic) 기반 피처 계산 모듈

입력 : date, name, gender_m_pct, gender_f_pct, age_10_pct, age_20_pct, age_30_pct, 
       age_40_pct, age_50_pct, age_60_pct 컬럼을 가진 DataFrame
출력 : 아래 14개 피처 컬럼이 추가된 DataFrame

  [성별 기반 (4개)]
    male_click_ratio: 남성 클릭 비율
    female_click_ratio: 여성 클릭 비율
    gender_click_skew: 성별 편중도
    gender_click_shift_7d: 최근 7일 남성 비율 변화

  [연령대 기반 (10개)]
    age10/20/30/40/50p_click_ratio: 각 연령대 비율
    young_click_ratio: 10~20대 합산
    mid_click_ratio: 30~40대 합산
    core_age_click_ratio: 핵심 연령대 비율
    age_click_entropy: 연령 분포 엔트로피
    age_click_shift_7d: 핵심 연령대 7일 비율 변화

사용법:
  from feature_engineering.demographic_features import compute_demographic_features
  df = compute_demographic_features(raw_df)
"""

import numpy as np
import pandas as pd

_EPS = 1e-8  # 엔트로피 계산용 작은 값


# ── 키워드별 피처 계산 ────────────────────────────────────────────────────────

def _features_per_name(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()

    # ── 성별 기반 (4개) ──────────────────────────────────────────────────────
    # 남성/여성 비율 (1일 전 데이터 사용, t-1)
    g["male_click_ratio"] = g["gender_m_pct"].shift(1)
    g["female_click_ratio"] = g["gender_f_pct"].shift(1)

    # 성별 편중도: 절대값(남성비율 - 여성비율)
    g["gender_click_skew"] = (g["male_click_ratio"] - g["female_click_ratio"]).abs()

    # 성별 변화: 최근 7일 남성 비율 변화
    g["gender_click_shift_7d"] = g["male_click_ratio"] - g["male_click_ratio"].shift(7)

    # ── 연령대 기반 (10개) ──────────────────────────────────────────────────
    # 각 연령대 비율 (t-1)
    g["age10_click_ratio"] = g["age_10_pct"].shift(1)
    g["age20_click_ratio"] = g["age_20_pct"].shift(1)
    g["age30_click_ratio"] = g["age_30_pct"].shift(1)
    g["age40_click_ratio"] = g["age_40_pct"].shift(1)

    # 50대 이상 합산
    age50_pct = g.get("age_50_pct", pd.Series(0, index=g.index))
    age60_pct = g.get("age_60_pct", pd.Series(0, index=g.index))
    g["age50p_click_ratio"] = (age50_pct + age60_pct).shift(1)

    # 연령대 그룹
    g["young_click_ratio"] = g["age10_click_ratio"] + g["age20_click_ratio"]
    g["mid_click_ratio"] = g["age30_click_ratio"] + g["age40_click_ratio"]

    # 핵심 연령대 (최대 비율)
    age_cols = [
        g["age10_click_ratio"],
        g["age20_click_ratio"],
        g["age30_click_ratio"],
        g["age40_click_ratio"],
        g["age50p_click_ratio"],
    ]
    g["core_age_click_ratio"] = pd.concat(age_cols, axis=1).max(axis=1)

    # 연령 분포 엔트로피: -Σ p·log(p+ε)
    def entropy(row):
        probs = row.dropna()
        probs = probs[probs > 0]
        if len(probs) == 0:
            return np.nan
        return -np.sum(probs * np.log(probs + _EPS))

    age_dist = pd.concat(
        [
            g["age10_click_ratio"],
            g["age20_click_ratio"],
            g["age30_click_ratio"],
            g["age40_click_ratio"],
            g["age50p_click_ratio"],
        ],
        axis=1,
    )
    g["age_click_entropy"] = age_dist.apply(entropy, axis=1)

    # 핵심 연령대 7일 변화
    g["age_click_shift_7d"] = g["core_age_click_ratio"] - g["core_age_click_ratio"].shift(7)

    return g


# ── 공개 API ──────────────────────────────────────────────────────────────────

def compute_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    인구통계 기반 피처 14개를 계산해 반환한다.

    Parameters
    ----------
    df : DataFrame
        최소 컬럼: date, name,
                  gender_m_pct, gender_f_pct,
                  age_10_pct, age_20_pct, age_30_pct, age_40_pct,
                  age_50_pct (또는 age_50_pct+age_60_pct)
        모든 값은 비율(0~1 또는 0~100, 정규화됨).

    Returns
    -------
    DataFrame
        키워드별 날짜 오름차순 정렬, 14개 피처 컬럼 추가.
    """
    required = {"date", "name", "gender_m_pct", "gender_f_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[demographic_features] 필수 컬럼 없음: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["name", "date"]).reset_index(drop=True)
    df = df.groupby("name", group_keys=False).apply(_features_per_name)
    return df.reset_index(drop=True)
