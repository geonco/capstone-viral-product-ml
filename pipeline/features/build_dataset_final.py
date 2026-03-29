"""
pipeline/features/build_dataset_final.py

최종 데이터셋 생성 스크립트 (Phase 1+2+3, 51개 피처)

입력:
    data/raw/search_absolute.csv
    data/raw/shopping_click_absolute.csv
    data/raw/sometrend_mention_long.csv
    data/raw/gender_m_pct.csv, gender_f_pct.csv
    data/raw/age_10_pct.csv ~ age_60_pct.csv (6개)

출력:
    data/processed/dataset_final.csv

피처 (51개):
    Phase 1 — 검색량 기반 10개 + 클릭수 기반 10개 + 결합 4개
    Phase 2 — SomeTrend 언급량 13개
    Phase 3 — Demographic 클릭 비율 14개

레이블 (2개):
    virality_score = future14_max / past60_mean  (성장 배율)
    peak_time      = argmax(search[t : t+13])    (0~13)

샘플 기간: 2022-05-01 ~ 2026-02-14 (예측 시점 기준)
window   : 60일 lookback + 14일 forward = 74일
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
DATASET_START  = pd.Timestamp("2022-05-01")
DATASET_END    = pd.Timestamp("2026-02-14")
LOOKBACK       = 60
FORWARD        = 14
MIN_VALID_DAYS = 30
EPS            = 1e-6


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────

def load_wide(path: Path) -> pd.DataFrame:
    """wide-format CSV 로드. Returns: DataFrame (index=keyword, columns=DatetimeIndex)"""
    df = pd.read_csv(path, encoding="utf-8-sig", index_col="keyword")
    df.columns = pd.to_datetime(df.columns)
    return df


def fill_nan(s: pd.Series) -> np.ndarray:
    """ffill → bfill → 0 순서로 NaN 처리 후 float64 numpy 배열 반환"""
    return s.ffill().bfill().fillna(0.0).values.astype(np.float64)


def _slope(arr: np.ndarray) -> float:
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    coef = np.polyfit(x, arr, 1)
    return float(coef[0])


def _growth(now: np.ndarray, prev: np.ndarray) -> float:
    return float(np.mean(now)) / (float(np.mean(prev)) + EPS) - 1.0


def _pos_14d(val: float, window14: np.ndarray) -> float:
    return val / (float(np.max(window14)) + EPS)


def _days_since_max(window14: np.ndarray) -> int:
    return 13 - int(np.argmax(window14))


# ── 피처·레이블 계산 ───────────────────────────────────────────────────────────

def compute_row(
    s_arr:    np.ndarray,
    c_arr:    np.ndarray,
    m_arr:    np.ndarray,
    gm_arr:   np.ndarray,
    gf_arr:   np.ndarray,
    a10_arr:  np.ndarray,
    a20_arr:  np.ndarray,
    a30_arr:  np.ndarray,
    a40_arr:  np.ndarray,
    a50p_arr: np.ndarray,
    t:        int,
    valid_mask_s: np.ndarray,
    valid_mask_c: np.ndarray,
) -> dict | None:
    """
    t 시점 1개 샘플의 피처와 레이블을 계산

    Parameters
    ----------
    s_arr    : 검색량 전체 시계열 (NaN fill 완료)
    c_arr    : 클릭수 전체 시계열 (NaN fill 완료)
    m_arr    : SomeTrend 언급량 전체 시계열 (NaN fill 완료, 없는 키워드는 0 배열)
    gm_arr   : 남성 클릭 비율 % 시계열
    gf_arr   : 여성 클릭 비율 % 시계열
    a10_arr  : 10대 클릭 비율 % 시계열
    a20_arr  : 20대 클릭 비율 % 시계열
    a30_arr  : 30대 클릭 비율 % 시계열
    a40_arr  : 40대 클릭 비율 % 시계열
    a50p_arr : 50대+ 클릭 비율 % 시계열 (= age_50_pct + age_60_pct)
    t        : 예측 시점 인덱스 (0-indexed)
    valid_mask_s : 원본 NaN 여부 마스크 (True=유효)
    valid_mask_c : 원본 NaN 여부 마스크 (True=유효)

    Returns None if 데이터 부족
    """
    lb_s = t - LOOKBACK
    fw_e = t + FORWARD

    if lb_s < 0 or fw_e > len(s_arr):
        return None
    if valid_mask_s[lb_s:t].sum() < MIN_VALID_DAYS:
        return None
    if valid_mask_c[lb_s:t].sum() < MIN_VALID_DAYS:
        return None

    s_lb = s_arr[lb_s:t]   # shape (60,)
    c_lb = c_arr[lb_s:t]
    m_lb = m_arr[lb_s:t]
    s_fw = s_arr[t:fw_e]   # shape (14,)

    # ── 서브 윈도우 ──
    s3   = s_lb[-3:];  s3p  = s_lb[-6:-3]
    s7   = s_lb[-7:];  s7p  = s_lb[-14:-7]
    s14  = s_lb[-14:]; s14p = s_lb[-28:-14]

    c3   = c_lb[-3:];  c3p  = c_lb[-6:-3]
    c7   = c_lb[-7:];  c7p  = c_lb[-14:-7]
    c14  = c_lb[-14:]; c14p = c_lb[-28:-14]

    m3   = m_lb[-3:];  m3p  = m_lb[-6:-3]
    m7   = m_lb[-7:];  m7p  = m_lb[-14:-7]
    m14  = m_lb[-14:]; m14p = m_lb[-28:-14]

    # ── Phase 1: 검색량 피처 (10개) ──
    s_ma_7d        = float(np.mean(s7))
    s_growth_3d    = _growth(s3,  s3p)
    s_growth_7d    = _growth(s7,  s7p)
    s_growth_14d   = _growth(s14, s14p)
    s_slope_7d     = _slope(s7)
    s_slope_14d    = _slope(s14)
    s_acceleration = s_growth_7d - s_growth_14d
    s_std_7d       = float(np.std(s7))
    s_pos_14d      = _pos_14d(s_lb[-1], s14)
    days_since_max = _days_since_max(s14)

    # ── Phase 1: 클릭수 피처 (10개) ──
    c_ma_7d              = float(np.mean(c7))
    c_growth_3d          = _growth(c3,  c3p)
    c_growth_7d          = _growth(c7,  c7p)
    c_growth_14d         = _growth(c14, c14p)
    c_slope_7d           = _slope(c7)
    c_slope_14d          = _slope(c14)
    c_acceleration       = c_growth_7d - c_growth_14d
    c_std_7d             = float(np.std(c7))
    c_pos_14d            = _pos_14d(c_lb[-1], c14)
    days_since_click_max = _days_since_max(c14)

    # ── Phase 1: 결합 피처 (4개) ──
    click_search_ratio   = c_ma_7d / (s_ma_7d + EPS)
    click_lead_signal_7d = c_growth_7d - s_growth_7d
    engagement_force     = c_ma_7d * s_slope_7d
    click_peak_gap       = c_pos_14d - s_pos_14d

    # ── Phase 2: SomeTrend 언급량 피처 (13개) ──
    m_ma_3d                = float(np.mean(m3))
    m_ma_7d                = float(np.mean(m7))
    m_growth_3d            = _growth(m3,  m3p)
    m_growth_7d            = _growth(m7,  m7p)
    m_growth_14d           = _growth(m14, m14p)
    m_acceleration         = m_growth_7d - m_growth_14d
    m_std_7d               = float(np.std(m7))
    m_pos_14d              = _pos_14d(m_lb[-1], m14)
    days_since_mention_max = _days_since_max(m14)
    mention_search_ratio   = m_ma_7d / (s_ma_7d + EPS)
    lead_signal_7d         = m_growth_7d - s_growth_7d
    viral_force            = m_ma_7d * s_slope_7d
    peak_gap               = m_pos_14d - s_pos_14d

    # ── Phase 3: Demographic 피처 (14개) ──
    # 단일 시점: t-1 (예측 직전 가장 최근 값)
    male_r   = float(gm_arr[t - 1])
    female_r = float(gf_arr[t - 1])
    a10_r    = float(a10_arr[t - 1])
    a20_r    = float(a20_arr[t - 1])
    a30_r    = float(a30_arr[t - 1])
    a40_r    = float(a40_arr[t - 1])
    a50p_r   = float(a50p_arr[t - 1])

    male_click_ratio      = male_r
    female_click_ratio    = female_r
    gender_click_skew     = abs(male_r - female_r)
    # 최근 7일간 남성 비율 변화 (t-1 기준 → t-7 기준)
    gender_click_shift_7d = male_r - float(gm_arr[t - 7])

    age10_click_ratio  = a10_r
    age20_click_ratio  = a20_r
    age30_click_ratio  = a30_r
    age40_click_ratio  = a40_r
    age50p_click_ratio = a50p_r
    young_click_ratio  = a10_r + a20_r
    mid_click_ratio    = a30_r + a40_r

    age_vals             = np.array([a10_r, a20_r, a30_r, a40_r, a50p_r])
    core_age_click_ratio = float(np.max(age_vals))
    core_age_idx         = int(np.argmax(age_vals))

    # entropy: 연령 분포 균등도 (높을수록 고른 분포)
    age_probs         = age_vals / (age_vals.sum() + EPS)
    age_click_entropy = float(-np.sum(age_probs * np.log(age_probs + EPS)))

    # 핵심 연령대 비율 7일 변화
    core_age_arrs    = [a10_arr, a20_arr, a30_arr, a40_arr, a50p_arr]
    age_click_shift_7d = (
        float(core_age_arrs[core_age_idx][t - 1])
        - float(core_age_arrs[core_age_idx][t - 7])
    )

    # ── 레이블 ──
    past60_mean    = float(np.mean(s_lb))
    future14_max   = float(np.max(s_fw))
    virality_score = future14_max / (past60_mean + EPS)
    peak_time      = int(np.argmax(s_fw))

    return {
        # Phase 1: 검색량
        "search_ma_7d":           s_ma_7d,
        "search_growth_3d":       s_growth_3d,
        "search_growth_7d":       s_growth_7d,
        "search_growth_14d":      s_growth_14d,
        "search_slope_7d":        s_slope_7d,
        "search_slope_14d":       s_slope_14d,
        "search_acceleration":    s_acceleration,
        "search_std_7d":          s_std_7d,
        "search_pos_14d":         s_pos_14d,
        "days_since_max_14d":     days_since_max,
        # Phase 1: 클릭수
        "click_ma_7d":            c_ma_7d,
        "click_growth_3d":        c_growth_3d,
        "click_growth_7d":        c_growth_7d,
        "click_growth_14d":       c_growth_14d,
        "click_slope_7d":         c_slope_7d,
        "click_slope_14d":        c_slope_14d,
        "click_acceleration":     c_acceleration,
        "click_std_7d":           c_std_7d,
        "click_pos_14d":          c_pos_14d,
        "days_since_click_max_14d": days_since_click_max,
        # Phase 1: 결합
        "click_search_ratio":     click_search_ratio,
        "click_lead_signal_7d":   click_lead_signal_7d,
        "engagement_force":       engagement_force,
        "click_peak_gap":         click_peak_gap,
        # Phase 2: SomeTrend
        "mention_ma_3d":               m_ma_3d,
        "mention_ma_7d":               m_ma_7d,
        "mention_growth_3d":           m_growth_3d,
        "mention_growth_7d":           m_growth_7d,
        "mention_growth_14d":          m_growth_14d,
        "mention_acceleration":        m_acceleration,
        "mention_std_7d":              m_std_7d,
        "mention_pos_14d":             m_pos_14d,
        "days_since_mention_max_14d":  days_since_mention_max,
        "mention_search_ratio":        mention_search_ratio,
        "lead_signal_7d":              lead_signal_7d,
        "viral_force":                 viral_force,
        "peak_gap":                    peak_gap,
        # Phase 3: Demographic
        "male_click_ratio":       male_click_ratio,
        "female_click_ratio":     female_click_ratio,
        "gender_click_skew":      gender_click_skew,
        "gender_click_shift_7d":  gender_click_shift_7d,
        "age10_click_ratio":      age10_click_ratio,
        "age20_click_ratio":      age20_click_ratio,
        "age30_click_ratio":      age30_click_ratio,
        "age40_click_ratio":      age40_click_ratio,
        "age50p_click_ratio":     age50p_click_ratio,
        "young_click_ratio":      young_click_ratio,
        "mid_click_ratio":        mid_click_ratio,
        "core_age_click_ratio":   core_age_click_ratio,
        "age_click_entropy":      age_click_entropy,
        "age_click_shift_7d":     age_click_shift_7d,
        # 레이블
        "virality_score":         round(virality_score, 6),
        "peak_time":              peak_time,
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("최종 데이터셋 생성 (Phase 1+2+3, 51개 피처)")
    print("=" * 60)

    # ── 로드 ──
    print("\n[1/4] CSV 로드 중...")
    search_df = load_wide(RAW_DIR / "search_absolute.csv")
    click_df  = load_wide(RAW_DIR / "shopping_click_absolute.csv")
    print(f"  search_absolute  : {search_df.shape[0]}개 키워드 × {search_df.shape[1]}일")
    print(f"  click_absolute   : {click_df.shape[0]}개 키워드 × {click_df.shape[1]}일")

    # SomeTrend long → wide pivot (컬럼 인덱스 2 = 합계)
    mention_raw = pd.read_csv(
        RAW_DIR / "sometrend_mention_long.csv", encoding="utf-8-sig"
    )
    mention_col = mention_raw.columns[2]
    mention_df  = mention_raw.pivot(index="keyword", columns="date", values=mention_col)
    mention_df.columns = pd.to_datetime(mention_df.columns)
    mention_kws = set(mention_df.index)
    print(f"  sometrend_mention: {mention_df.shape[0]}개 키워드 × {mention_df.shape[1]}일")

    # Demographic wide 로드
    gm_df  = load_wide(RAW_DIR / "gender_m_pct.csv")
    gf_df  = load_wide(RAW_DIR / "gender_f_pct.csv")
    a10_df = load_wide(RAW_DIR / "age_10_pct.csv")
    a20_df = load_wide(RAW_DIR / "age_20_pct.csv")
    a30_df = load_wide(RAW_DIR / "age_30_pct.csv")
    a40_df = load_wide(RAW_DIR / "age_40_pct.csv")
    a50_df = load_wide(RAW_DIR / "age_50_pct.csv")
    a60_df = load_wide(RAW_DIR / "age_60_pct.csv")
    print(f"  demographic      : 8개 파일 로드 완료")

    # ── 공통 키워드 / 날짜 ──
    common_kw = search_df.index.intersection(click_df.index)
    all_dates = (
        search_df.columns
        .intersection(click_df.columns)
        .intersection(gm_df.columns)
        .sort_values()
    )
    print(f"\n  공통 키워드: {len(common_kw)}개")
    print(f"  공통 날짜  : {len(all_dates)}일  "
          f"({all_dates[0].date()} ~ {all_dates[-1].date()})")

    pred_mask  = (all_dates >= DATASET_START) & (all_dates <= DATASET_END)
    pred_dates = all_dates[pred_mask]
    pred_idxs  = np.where(pred_mask)[0]
    print(f"  예측 시점  : {len(pred_dates)}일  "
          f"({pred_dates[0].date()} ~ {pred_dates[-1].date()})")

    n_mention_missing = len(common_kw) - len(common_kw.intersection(mention_kws))
    print(f"\n  SomeTrend 미보유 키워드: {n_mention_missing}개 → 0으로 채움")

    # ── 피처 계산 ──
    print("\n[2/4] 피처 계산 중...")
    records   = []
    n_kw      = len(common_kw)
    n_skipped = 0

    for ki, kw in enumerate(common_kw, 1):
        if ki % 100 == 0 or ki == n_kw:
            print(f"  {ki}/{n_kw} 키워드 완료  (현재 샘플 수: {len(records):,})")

        s_raw   = search_df.loc[kw, all_dates]
        c_raw   = click_df.loc[kw, all_dates]
        valid_s = ~s_raw.isna().values
        valid_c = ~c_raw.isna().values
        s_arr   = fill_nan(s_raw)
        c_arr   = fill_nan(c_raw)

        # SomeTrend: 없는 키워드는 0 배열 (언급량이 실제 0인 것과 동일 처리)
        if kw in mention_kws:
            m_arr = fill_nan(mention_df.loc[kw].reindex(all_dates))
        else:
            m_arr = np.zeros(len(all_dates), dtype=np.float64)

        # Demographic: all_dates 기준으로 reindex 후 fill
        gm_arr   = fill_nan(gm_df.loc[kw].reindex(all_dates))
        gf_arr   = fill_nan(gf_df.loc[kw].reindex(all_dates))
        a10_arr  = fill_nan(a10_df.loc[kw].reindex(all_dates))
        a20_arr  = fill_nan(a20_df.loc[kw].reindex(all_dates))
        a30_arr  = fill_nan(a30_df.loc[kw].reindex(all_dates))
        a40_arr  = fill_nan(a40_df.loc[kw].reindex(all_dates))
        a50_arr  = fill_nan(a50_df.loc[kw].reindex(all_dates))
        a60_arr  = fill_nan(a60_df.loc[kw].reindex(all_dates))
        a50p_arr = a50_arr + a60_arr   # 50대+ = 50대 + 60대

        for t_idx in pred_idxs:
            row = compute_row(
                s_arr, c_arr, m_arr,
                gm_arr, gf_arr,
                a10_arr, a20_arr, a30_arr, a40_arr, a50p_arr,
                t_idx, valid_s, valid_c,
            )
            if row is None:
                n_skipped += 1
                continue
            row["keyword"] = kw
            row["date"]    = all_dates[t_idx].date()
            records.append(row)

    print(f"\n  총 샘플: {len(records):,}  (제외: {n_skipped:,})")

    # ── DataFrame 구성 ──
    print("\n[3/4] DataFrame 구성 중...")
    df = pd.DataFrame(records)

    FEATURE_COLS = [
        # Phase 1: 검색량 (10)
        "search_ma_7d", "search_growth_3d", "search_growth_7d", "search_growth_14d",
        "search_slope_7d", "search_slope_14d", "search_acceleration", "search_std_7d",
        "search_pos_14d", "days_since_max_14d",
        # Phase 1: 클릭수 (10)
        "click_ma_7d", "click_growth_3d", "click_growth_7d", "click_growth_14d",
        "click_slope_7d", "click_slope_14d", "click_acceleration", "click_std_7d",
        "click_pos_14d", "days_since_click_max_14d",
        # Phase 1: 결합 (4)
        "click_search_ratio", "click_lead_signal_7d", "engagement_force", "click_peak_gap",
        # Phase 2: SomeTrend (13)
        "mention_ma_3d", "mention_ma_7d", "mention_growth_3d", "mention_growth_7d",
        "mention_growth_14d", "mention_acceleration", "mention_std_7d", "mention_pos_14d",
        "days_since_mention_max_14d", "mention_search_ratio", "lead_signal_7d",
        "viral_force", "peak_gap",
        # Phase 3: Demographic (14)
        "male_click_ratio", "female_click_ratio", "gender_click_skew", "gender_click_shift_7d",
        "age10_click_ratio", "age20_click_ratio", "age30_click_ratio", "age40_click_ratio",
        "age50p_click_ratio", "young_click_ratio", "mid_click_ratio",
        "core_age_click_ratio", "age_click_entropy", "age_click_shift_7d",
    ]
    LABEL_COLS = ["virality_score", "peak_time"]

    assert len(FEATURE_COLS) == 51, f"피처 수 불일치: {len(FEATURE_COLS)}"

    df = df[["keyword", "date"] + FEATURE_COLS + LABEL_COLS]
    df = df.sort_values(["keyword", "date"]).reset_index(drop=True)

    # ── 저장 ──
    print("\n[4/4] 저장 중...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "dataset_final.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {out_path}")

    # ── 요약 통계 ──
    print("\n" + "=" * 60)
    print("데이터셋 요약")
    print("=" * 60)
    print(f"shape          : {df.shape}")
    print(f"키워드 수      : {df['keyword'].nunique()}")
    print(f"날짜 범위      : {df['date'].min()} ~ {df['date'].max()}")
    print(f"피처 수        : {len(FEATURE_COLS)}")
    print(f"\nvirality_score:\n{df['virality_score'].describe().round(4)}")
    print(f"\npeak_time 분포 (0~13):\n{df['peak_time'].value_counts().sort_index().to_string()}")

    nan_counts = df[FEATURE_COLS + LABEL_COLS].isna().sum()
    if nan_counts.any():
        print(f"\n[경고] NaN 잔존:\n{nan_counts[nan_counts > 0]}")
    else:
        print("\n[OK] NaN 없음")


if __name__ == "__main__":
    main()
