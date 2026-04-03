"""
pipeline/features/build_dataset_v2.py

v2 데이터셋 생성 스크립트

v1(dataset_final.csv)과의 차이:
  - 변화량 피처 윈도우 확장: 3/7/14d → 3/5/7/14/30d (growth, slope, ma, std, pos, days_since_max)
  - acceleration 세분화: short(3d vs 7d), mid(7d vs 14d), long(14d vs 30d)
  - mention slope 추가 (v1에 없었음)
  - 피처 수: 51개 → 107개
  - 60일 품질 필터 기준 변경: NaN 기준 → 0 기준 (search 신호 기준, 0이 30개 이상이면 제외)
  - 레이블 추가: log_virality_score = log1p(virality_score)

입력:
    data/raw/search_absolute.csv
    data/raw/shopping_click_absolute.csv
    data/raw/sometrend_mention_long.csv
    data/raw/gender_m_pct.csv, gender_f_pct.csv
    data/raw/age_10_pct.csv ~ age_60_pct.csv (6개)

출력:
    data/processed/dataset_v2.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
RAW_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ── 하이퍼파라미터 ────────────────────────────────────────────────────────────
DATASET_START    = pd.Timestamp("2022-05-01")
DATASET_END      = pd.Timestamp("2026-02-14")
LOOKBACK         = 60
FORWARD          = 14
MIN_NONZERO_DAYS = LOOKBACK // 2   # search 60일 중 0이 아닌 값 최소 30개
EPS              = 1e-6


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────

def load_wide(path: Path) -> pd.DataFrame:
    """wide-format CSV 로드. Returns: DataFrame (index=keyword, columns=DatetimeIndex)"""
    df = pd.read_csv(path, encoding="utf-8-sig", index_col="keyword")
    df.columns = pd.to_datetime(df.columns)
    return df


def fill_nan(s: pd.Series) -> np.ndarray:
    """ffill → bfill → 0 순서로 NaN 처리 후 float64 numpy 배열 반환"""
    return s.ffill().bfill().fillna(0.0).values.astype(np.float64)


def _growth(now: np.ndarray, prev: np.ndarray) -> float:
    """mean(recent) / mean(prev) − 1"""
    return float(np.mean(now)) / (float(np.mean(prev)) + EPS) - 1.0


def _slope(arr: np.ndarray) -> float:
    """선형회귀 기울기. 데이터 2개 미만이면 0 반환"""
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    return float(np.polyfit(x, arr, 1)[0])


def _days_since_max(window: np.ndarray) -> int:
    """최근 n일 내 max 이후 경과일 (오늘이 max면 0)"""
    n = len(window)
    return (n - 1) - int(np.argmax(window))


# ── 신호별 피처 27개 계산 ─────────────────────────────────────────────────────

def _signal_features(arr_lb: np.ndarray, prefix: str) -> dict:
    """
    60일 lookback 배열에서 변화량 관련 피처 27개 계산.

    arr_lb는 길이 60의 float64 배열로, 인덱스 0이 60일 전, 인덱스 -1이 예측 직전(t-1)이다.
    prefix는 'search', 'click', 'mention' 중 하나로 피처 이름 앞에 붙인다.

    피처 구성은 다음과 같다.
    이동평균(ma) 5개: 3/5/7/14/30일 평균. 단기~중기 수준 파악용.
    성장률(growth) 5개: 직전 n일 평균 / 이전 n일 평균 - 1. 윈도우 3/5/7/14/30d.
      예) growth_5d = mean(arr[-5:]) / mean(arr[-10:-5]) - 1
    기울기(slope) 5개: 직전 n일 선형회귀 기울기. 윈도우 3/5/7/14/30d.
    표준편차(std) 3개: 직전 7/14/30일 표준편차. 변동성 파악용.
    위치(pos) 3개: 현재값 / 직전 n일 최고점. 윈도우 7/14/30d.
      1.0이면 n일 최고점에 위치, 0에 가까울수록 최고점 대비 낮은 위치.
    최고점 경과일(days_since_max) 3개: 직전 7/14/30일 내 최고점으로부터 경과일.
      0이면 오늘이 최고점.
    가속도(accel) 3개: growth 차이로 가속도 측정.
      accel_short = growth_3d - growth_7d  (초단기 vs 단기)
      accel_mid   = growth_7d - growth_14d (단기 vs 중기, v1 acceleration에 해당)
      accel_long  = growth_14d - growth_30d (중기 vs 장기)
    """
    p = prefix

    # 서브윈도우: (최근 n일, 이전 n일) 쌍
    w3   = arr_lb[-3:];  w3p  = arr_lb[-6:-3]
    w5   = arr_lb[-5:];  w5p  = arr_lb[-10:-5]
    w7   = arr_lb[-7:];  w7p  = arr_lb[-14:-7]
    w14  = arr_lb[-14:]; w14p = arr_lb[-28:-14]
    w30  = arr_lb[-30:]; w30p = arr_lb[:30]      # 60일 lookback 전반부(oldest 30)

    # growth
    g3  = _growth(w3,  w3p)
    g5  = _growth(w5,  w5p)
    g7  = _growth(w7,  w7p)
    g14 = _growth(w14, w14p)
    g30 = _growth(w30, w30p)

    # slope
    sl3  = _slope(w3)
    sl5  = _slope(w5)
    sl7  = _slope(w7)
    sl14 = _slope(w14)
    sl30 = _slope(w30)

    cur = arr_lb[-1]   # 현재값 (t-1)

    return {
        # 이동 평균
        f"{p}_ma_3d":  float(np.mean(w3)),
        f"{p}_ma_5d":  float(np.mean(w5)),
        f"{p}_ma_7d":  float(np.mean(w7)),
        f"{p}_ma_14d": float(np.mean(w14)),
        f"{p}_ma_30d": float(np.mean(w30)),
        # 성장률
        f"{p}_growth_3d":  g3,
        f"{p}_growth_5d":  g5,
        f"{p}_growth_7d":  g7,
        f"{p}_growth_14d": g14,
        f"{p}_growth_30d": g30,
        # 기울기
        f"{p}_slope_3d":   sl3,
        f"{p}_slope_5d":   sl5,
        f"{p}_slope_7d":   sl7,
        f"{p}_slope_14d":  sl14,
        f"{p}_slope_30d":  sl30,
        # 표준편차
        f"{p}_std_7d":  float(np.std(w7)),
        f"{p}_std_14d": float(np.std(w14)),
        f"{p}_std_30d": float(np.std(w30)),
        # 현재값의 n일 최고점 대비 위치
        f"{p}_pos_7d":  cur / (float(np.max(w7))  + EPS),
        f"{p}_pos_14d": cur / (float(np.max(w14)) + EPS),
        f"{p}_pos_30d": cur / (float(np.max(w30)) + EPS),
        # n일 내 최고점으로부터 경과일
        f"days_since_{p}_max_7d":  _days_since_max(w7),
        f"days_since_{p}_max_14d": _days_since_max(w14),
        f"days_since_{p}_max_30d": _days_since_max(w30),
        # 가속도
        f"{p}_accel_short": g3  - g7,    # 초단기 가속도 (3d vs 7d)
        f"{p}_accel_mid":   g7  - g14,   # 단기 가속도   (7d vs 14d)
        f"{p}_accel_long":  g14 - g30,   # 중기 가속도   (14d vs 30d)
    }


# ── 피처·레이블 계산 ───────────────────────────────────────────────────────────

def compute_row_v2(
    s_arr:     np.ndarray,
    c_arr:     np.ndarray,
    m_arr:     np.ndarray,
    gm_arr:    np.ndarray,
    gf_arr:    np.ndarray,
    a10_arr:   np.ndarray,
    a20_arr:   np.ndarray,
    a30_arr:   np.ndarray,
    a40_arr:   np.ndarray,
    a50p_arr:  np.ndarray,
    t:         int,
    s_invalid: np.ndarray,
) -> dict | None:
    """
    t 시점 1개 샘플의 피처(107개)와 레이블을 계산.

    s_arr, c_arr, m_arr는 각각 검색량·클릭수·언급량 전체 시계열(NaN fill 완료)이다.
    gm_arr~a50p_arr는 demographic 비율 시계열이다.
    t는 예측 시점 인덱스(0-indexed)이다.
    s_invalid는 fill_nan 적용 전 원본 기준 invalid 마스크(NaN이거나 0이면 True)이다.

    품질 필터: 60일 lookback의 앞 30일(가장 오래된 절반)이 모두 invalid(원본 NaN 또는 0)이면 None 반환.
    키워드가 해당 기간에 아직 존재하지 않았거나 검색량이 전혀 없었던 경우를 제거하는 목적.
    """
    lb_s = t - LOOKBACK
    fw_e = t + FORWARD

    if lb_s < 0 or fw_e > len(s_arr):
        return None

    # 품질 필터 — 60일 lookback 앞 30일이 모두 0이거나 NaN이면 제외
    if s_invalid[lb_s : lb_s + 30].all():
        return None

    s_lb = s_arr[lb_s:t]   # shape (60,)

    c_lb = c_arr[lb_s:t]
    m_lb = m_arr[lb_s:t]
    s_fw = s_arr[t:fw_e]   # shape (14,)

    # ── 신호별 피처 (각 27개) ──
    sf = _signal_features(s_lb, "search")
    cf = _signal_features(c_lb, "click")
    mf = _signal_features(m_lb, "mention")

    # ── search-click 결합 피처 (6개) ──
    # click_search_ratio: 검색 대비 클릭 전환 수준 (7일 이동평균 기준)
    # click_lead_Nd: 클릭 성장률 - 검색 성장률. 양수면 클릭이 검색보다 먼저 반응
    # engage_force_Nd: 클릭 평균 × 검색 기울기. 클릭 강도와 검색 모멘텀의 결합
    # click_peak_gap: 클릭 vs 검색 peak 위치 차이 (14일 기준)
    combined_sc = {
        "click_search_ratio": cf["click_ma_7d"]     / (sf["search_ma_7d"] + EPS),
        "click_lead_7d":      cf["click_growth_7d"]  - sf["search_growth_7d"],
        "click_lead_30d":     cf["click_growth_30d"] - sf["search_growth_30d"],
        "engage_force_7d":    cf["click_ma_7d"]     * sf["search_slope_7d"],
        "engage_force_30d":   cf["click_ma_30d"]    * sf["search_slope_30d"],
        "click_peak_gap":     cf["click_pos_14d"]    - sf["search_pos_14d"],
    }

    # ── mention-search 결합 피처 (6개) ──
    # mention_search_ratio: 검색량 대비 언급량 비율 (7일 기준)
    # mention_lead_Nd: 언급 성장률 - 검색 성장률. 양수면 SNS 버즈가 검색 앞서는 신호
    # viral_force_Nd: 언급 평균 × 검색 기울기. 버즈 강도와 검색 모멘텀의 결합
    # mention_peak_gap: 언급 vs 검색 peak 위치 차이 (14일 기준)
    combined_ms = {
        "mention_search_ratio": mf["mention_ma_7d"]     / (sf["search_ma_7d"] + EPS),
        "mention_lead_7d":      mf["mention_growth_7d"]  - sf["search_growth_7d"],
        "mention_lead_30d":     mf["mention_growth_30d"] - sf["search_growth_30d"],
        "viral_force_7d":       mf["mention_ma_7d"]     * sf["search_slope_7d"],
        "viral_force_30d":      mf["mention_ma_30d"]    * sf["search_slope_30d"],
        "mention_peak_gap":     mf["mention_pos_14d"]    - sf["search_pos_14d"],
    }

    # ── Phase 3: Demographic 피처 (14개, v1과 동일) ──
    male_r   = float(gm_arr[t - 1])
    female_r = float(gf_arr[t - 1])
    a10_r    = float(a10_arr[t - 1])
    a20_r    = float(a20_arr[t - 1])
    a30_r    = float(a30_arr[t - 1])
    a40_r    = float(a40_arr[t - 1])
    a50p_r   = float(a50p_arr[t - 1])

    age_vals             = np.array([a10_r, a20_r, a30_r, a40_r, a50p_r])
    core_age_idx         = int(np.argmax(age_vals))
    core_age_click_ratio = float(np.max(age_vals))
    age_probs            = age_vals / (age_vals.sum() + EPS)
    age_click_entropy    = float(-np.sum(age_probs * np.log(age_probs + EPS)))
    core_age_arrs        = [a10_arr, a20_arr, a30_arr, a40_arr, a50p_arr]
    age_click_shift_7d   = (
        float(core_age_arrs[core_age_idx][t - 1])
        - float(core_age_arrs[core_age_idx][t - 7])
    )

    demographic = {
        "male_click_ratio":      male_r,
        "female_click_ratio":    female_r,
        "gender_click_skew":     abs(male_r - female_r),
        "gender_click_shift_7d": male_r - float(gm_arr[t - 7]),
        "age10_click_ratio":     a10_r,
        "age20_click_ratio":     a20_r,
        "age30_click_ratio":     a30_r,
        "age40_click_ratio":     a40_r,
        "age50p_click_ratio":    a50p_r,
        "young_click_ratio":     a10_r + a20_r,
        "mid_click_ratio":       a30_r + a40_r,
        "core_age_click_ratio":  core_age_click_ratio,
        "age_click_entropy":     age_click_entropy,
        "age_click_shift_7d":    age_click_shift_7d,
    }

    # ── 레이블 ──
    past60_mean        = float(np.mean(s_lb))
    future14_max       = float(np.max(s_fw))
    virality_score     = future14_max / (past60_mean + EPS)
    peak_time          = int(np.argmax(s_fw))
    log_virality_score = float(np.log1p(virality_score))

    row = {}
    row.update(sf)
    row.update(cf)
    row.update(combined_sc)
    row.update(mf)
    row.update(combined_ms)
    row.update(demographic)
    row["virality_score"]     = round(virality_score, 6)
    row["log_virality_score"] = round(log_virality_score, 6)
    row["peak_time"]          = peak_time

    return row


# ── 피처 컬럼 순서 정의 ────────────────────────────────────────────────────────

FEATURE_COLS = [
    # search (27)
    "search_ma_3d", "search_ma_5d", "search_ma_7d", "search_ma_14d", "search_ma_30d",
    "search_growth_3d", "search_growth_5d", "search_growth_7d", "search_growth_14d", "search_growth_30d",
    "search_slope_3d", "search_slope_5d", "search_slope_7d", "search_slope_14d", "search_slope_30d",
    "search_std_7d", "search_std_14d", "search_std_30d",
    "search_pos_7d", "search_pos_14d", "search_pos_30d",
    "days_since_search_max_7d", "days_since_search_max_14d", "days_since_search_max_30d",
    "search_accel_short", "search_accel_mid", "search_accel_long",
    # click (27)
    "click_ma_3d", "click_ma_5d", "click_ma_7d", "click_ma_14d", "click_ma_30d",
    "click_growth_3d", "click_growth_5d", "click_growth_7d", "click_growth_14d", "click_growth_30d",
    "click_slope_3d", "click_slope_5d", "click_slope_7d", "click_slope_14d", "click_slope_30d",
    "click_std_7d", "click_std_14d", "click_std_30d",
    "click_pos_7d", "click_pos_14d", "click_pos_30d",
    "days_since_click_max_7d", "days_since_click_max_14d", "days_since_click_max_30d",
    "click_accel_short", "click_accel_mid", "click_accel_long",
    # search-click 결합 (6)
    "click_search_ratio",
    "click_lead_7d", "click_lead_30d",
    "engage_force_7d", "engage_force_30d",
    "click_peak_gap",
    # mention (27)
    "mention_ma_3d", "mention_ma_5d", "mention_ma_7d", "mention_ma_14d", "mention_ma_30d",
    "mention_growth_3d", "mention_growth_5d", "mention_growth_7d", "mention_growth_14d", "mention_growth_30d",
    "mention_slope_3d", "mention_slope_5d", "mention_slope_7d", "mention_slope_14d", "mention_slope_30d",
    "mention_std_7d", "mention_std_14d", "mention_std_30d",
    "mention_pos_7d", "mention_pos_14d", "mention_pos_30d",
    "days_since_mention_max_7d", "days_since_mention_max_14d", "days_since_mention_max_30d",
    "mention_accel_short", "mention_accel_mid", "mention_accel_long",
    # mention-search 결합 (6)
    "mention_search_ratio",
    "mention_lead_7d", "mention_lead_30d",
    "viral_force_7d", "viral_force_30d",
    "mention_peak_gap",
    # demographic (14)
    "male_click_ratio", "female_click_ratio", "gender_click_skew", "gender_click_shift_7d",
    "age10_click_ratio", "age20_click_ratio", "age30_click_ratio", "age40_click_ratio",
    "age50p_click_ratio", "young_click_ratio", "mid_click_ratio",
    "core_age_click_ratio", "age_click_entropy", "age_click_shift_7d",
]

LABEL_COLS = ["virality_score", "log_virality_score", "peak_time"]

assert len(FEATURE_COLS) == 107, f"피처 수 불일치: {len(FEATURE_COLS)}"


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("v2 데이터셋 생성 (107개 피처)")
    print("=" * 60)

    # ── 로드 ──
    print("\n[1/4] CSV 로드 중...")
    search_df = load_wide(RAW_DIR / "search_absolute.csv")
    click_df  = load_wide(RAW_DIR / "shopping_click_absolute.csv")
    print(f"  search_absolute  : {search_df.shape[0]}개 키워드 × {search_df.shape[1]}일")
    print(f"  click_absolute   : {click_df.shape[0]}개 키워드 × {click_df.shape[1]}일")

    mention_raw = pd.read_csv(RAW_DIR / "sometrend_mention_long.csv", encoding="utf-8-sig")
    mention_col = mention_raw.columns[2]
    mention_df  = mention_raw.pivot(index="keyword", columns="date", values=mention_col)
    mention_df.columns = pd.to_datetime(mention_df.columns)
    mention_kws = set(mention_df.index)
    print(f"  sometrend_mention: {mention_df.shape[0]}개 키워드 × {mention_df.shape[1]}일")

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
    all_dates  = (
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

        s_raw = search_df.loc[kw, all_dates]
        # fill_nan 전 원본 기준 invalid 마스크: NaN이거나 0이면 True
        s_invalid = (s_raw.isna() | (s_raw == 0)).values
        s_arr = fill_nan(s_raw)
        c_arr = fill_nan(click_df.loc[kw, all_dates])

        if kw in mention_kws:
            m_arr = fill_nan(mention_df.loc[kw].reindex(all_dates))
        else:
            m_arr = np.zeros(len(all_dates), dtype=np.float64)

        gm_arr   = fill_nan(gm_df.loc[kw].reindex(all_dates))
        gf_arr   = fill_nan(gf_df.loc[kw].reindex(all_dates))
        a10_arr  = fill_nan(a10_df.loc[kw].reindex(all_dates))
        a20_arr  = fill_nan(a20_df.loc[kw].reindex(all_dates))
        a30_arr  = fill_nan(a30_df.loc[kw].reindex(all_dates))
        a40_arr  = fill_nan(a40_df.loc[kw].reindex(all_dates))
        a50_arr  = fill_nan(a50_df.loc[kw].reindex(all_dates))
        a60_arr  = fill_nan(a60_df.loc[kw].reindex(all_dates))
        a50p_arr = a50_arr + a60_arr

        for t_idx in pred_idxs:
            row = compute_row_v2(
                s_arr, c_arr, m_arr,
                gm_arr, gf_arr,
                a10_arr, a20_arr, a30_arr, a40_arr, a50p_arr,
                t_idx,
                s_invalid,
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
    df = df[["keyword", "date"] + FEATURE_COLS + LABEL_COLS]
    df = df.sort_values(["keyword", "date"]).reset_index(drop=True)

    # ── 저장 ──
    print("\n[4/4] 저장 중...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "dataset_v2.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  저장 완료: {out_path}")

    # ── 요약 통계 ──
    print("\n" + "=" * 60)
    print("데이터셋 요약")
    print("=" * 60)
    print(f"shape               : {df.shape}")
    print(f"키워드 수           : {df['keyword'].nunique()}")
    print(f"날짜 범위           : {df['date'].min()} ~ {df['date'].max()}")
    print(f"피처 수             : {len(FEATURE_COLS)}")
    print(f"\nvirality_score:\n{df['virality_score'].describe().round(4)}")
    print(f"\nlog_virality_score:\n{df['log_virality_score'].describe().round(4)}")
    print(f"\npeak_time 분포 (0~13):\n{df['peak_time'].value_counts().sort_index().to_string()}")

    nan_counts = df[FEATURE_COLS + LABEL_COLS].isna().sum()
    if nan_counts.any():
        print(f"\n[경고] NaN 잔존:\n{nan_counts[nan_counts > 0]}")
    else:
        print("\n[OK] NaN 없음")


if __name__ == "__main__":
    main()
