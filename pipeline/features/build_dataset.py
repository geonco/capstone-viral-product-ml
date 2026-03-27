"""
pipeline/features/build_dataset.py

Phase 1 데이터셋 생성 스크립트.

입력:
    data/raw/search/absolute.csv
    data/raw/shopping_click/absolute.csv

출력:
    data/processed/dataset_phase1.csv

피처 (24개):
    검색량 기반 10개 + 클릭수 기반 10개 + 결합 4개

레이블 (2개):
    virality_score = future14_max / past60_mean  (성장 배율)
    peak_time      = argmax(search[t : t+14])    (0~13)

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
LOOKBACK       = 60     # 피처 계산용 lookback 일수
FORWARD        = 14     # 레이블 계산용 forward 일수
MIN_VALID_DAYS = 30     # lookback 내 원본 유효값 최소 수 (fill 이전 기준)
EPS            = 1e-6   # 0 나누기 방지


# ── 유틸 함수 ─────────────────────────────────────────────────────────────────

def load_wide(path: Path) -> pd.DataFrame:
    """
    wide-format CSV 로드.
    Returns: DataFrame  (index=keyword, columns=DatetimeIndex)
    """
    df = pd.read_csv(path, encoding="utf-8-sig", index_col="keyword")
    df.columns = pd.to_datetime(df.columns)
    return df


def fill_nan(s: pd.Series) -> np.ndarray:
    """ffill → bfill → 0 순서로 NaN 처리 후 float64 numpy 배열 반환."""
    return s.ffill().bfill().fillna(0.0).values.astype(np.float64)


def _slope(arr: np.ndarray) -> float:
    """최소자승법 기울기. 원소 수 < 2 이면 0 반환."""
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    # polyfit이 linregress보다 빠름 (scipy import 없이도 가능)
    coef = np.polyfit(x, arr, 1)
    return float(coef[0])


def _growth(now: np.ndarray, prev: np.ndarray) -> float:
    """mean(now) / (mean(prev) + ε) - 1"""
    return float(np.mean(now)) / (float(np.mean(prev)) + EPS) - 1.0


def _pos_14d(val: float, window14: np.ndarray) -> float:
    """현재값 / (14일 window 최대값 + ε)"""
    return val / (float(np.max(window14)) + EPS)


def _days_since_max(window14: np.ndarray) -> int:
    """
    14일 window 내 최대값으로부터 경과일.
    window14[0] = 14일 전, window14[13] = 어제(t-1)
    argmax=13 → 0일 경과, argmax=0 → 13일 경과
    """
    return 13 - int(np.argmax(window14))


# ── 피처·레이블 계산 ───────────────────────────────────────────────────────────

def compute_row(
    s_arr: np.ndarray,
    c_arr: np.ndarray,
    t: int,
    valid_mask_s: np.ndarray,
    valid_mask_c: np.ndarray,
) -> dict | None:
    """
    t 시점 1개 샘플의 피처와 레이블을 계산한다.

    Parameters
    ----------
    s_arr        : 검색량 전체 시계열 (NaN fill 완료)
    c_arr        : 클릭수 전체 시계열 (NaN fill 완료)
    t            : 예측 시점 인덱스 (0-indexed)
    valid_mask_s : 원본 NaN 여부 마스크 (True=유효)
    valid_mask_c : 원본 NaN 여부 마스크 (True=유효)

    Returns None if 데이터 부족.
    """
    lb_s = t - LOOKBACK
    fw_e = t + FORWARD

    # ── 범위 검사 ──
    if lb_s < 0 or fw_e > len(s_arr):
        return None

    # ── 원본 유효값 충분한지 확인 (fill 이전 기준) ──
    if valid_mask_s[lb_s:t].sum() < MIN_VALID_DAYS:
        return None
    if valid_mask_c[lb_s:t].sum() < MIN_VALID_DAYS:
        return None

    # ── lookback / forward 슬라이스 ──
    s_lb = s_arr[lb_s:t]   # shape (60,)   s_lb[59] = t-1 (어제)
    c_lb = c_arr[lb_s:t]
    s_fw = s_arr[t:fw_e]   # shape (14,)   s_fw[0]  = t (오늘)

    # ── 서브 윈도우 ──
    # 검색량
    s3   = s_lb[-3:]        # 최근 3일  [t-3, t-2, t-1]
    s3p  = s_lb[-6:-3]      # 이전 3일  [t-6, t-5, t-4]
    s7   = s_lb[-7:]        # 최근 7일
    s7p  = s_lb[-14:-7]     # 이전 7일
    s14  = s_lb[-14:]       # 최근 14일
    s14p = s_lb[-28:-14]    # 이전 14일

    # 클릭수
    c3   = c_lb[-3:]
    c3p  = c_lb[-6:-3]
    c7   = c_lb[-7:]
    c7p  = c_lb[-14:-7]
    c14  = c_lb[-14:]
    c14p = c_lb[-28:-14]

    # ── 검색량 피처 (10개) ──
    s_ma_7d          = float(np.mean(s7))
    s_growth_3d      = _growth(s3,  s3p)
    s_growth_7d      = _growth(s7,  s7p)
    s_growth_14d     = _growth(s14, s14p)
    s_slope_7d       = _slope(s7)
    s_slope_14d      = _slope(s14)
    s_acceleration   = s_growth_7d - s_growth_14d
    s_std_7d         = float(np.std(s7))
    s_pos_14d        = _pos_14d(s_lb[-1], s14)
    days_since_max   = _days_since_max(s14)

    # ── 클릭수 피처 (10개) ──
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

    # ── 결합 피처 (4개) ──
    click_search_ratio   = c_ma_7d / (s_ma_7d + EPS)
    click_lead_signal_7d = c_growth_7d - s_growth_7d
    engagement_force     = c_ma_7d * s_slope_7d
    click_peak_gap       = c_pos_14d - s_pos_14d

    # ── 레이블 ──
    past60_mean    = float(np.mean(s_lb))
    future14_max   = float(np.max(s_fw))
    virality_score = future14_max / (past60_mean + EPS)
    peak_time      = int(np.argmax(s_fw))    # 0~13

    return {
        # 검색량
        "search_ma_7d":          s_ma_7d,
        "search_growth_3d":      s_growth_3d,
        "search_growth_7d":      s_growth_7d,
        "search_growth_14d":     s_growth_14d,
        "search_slope_7d":       s_slope_7d,
        "search_slope_14d":      s_slope_14d,
        "search_acceleration":   s_acceleration,
        "search_std_7d":         s_std_7d,
        "search_pos_14d":        s_pos_14d,
        "days_since_max_14d":    days_since_max,
        # 클릭수
        "click_ma_7d":               c_ma_7d,
        "click_growth_3d":           c_growth_3d,
        "click_growth_7d":           c_growth_7d,
        "click_growth_14d":          c_growth_14d,
        "click_slope_7d":            c_slope_7d,
        "click_slope_14d":           c_slope_14d,
        "click_acceleration":        c_acceleration,
        "click_std_7d":              c_std_7d,
        "click_pos_14d":             c_pos_14d,
        "days_since_click_max_14d":  days_since_click_max,
        # 결합
        "click_search_ratio":    click_search_ratio,
        "click_lead_signal_7d":  click_lead_signal_7d,
        "engagement_force":      engagement_force,
        "click_peak_gap":        click_peak_gap,
        # 레이블
        "virality_score":        round(virality_score, 6),
        "peak_time":             peak_time,
    }


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Phase 1 데이터셋 생성")
    print("=" * 60)

    # ── 로드 ──
    print("\n[1/4] CSV 로드 중...")
    search_df = load_wide(RAW_DIR / "search" / "absolute.csv")
    click_df  = load_wide(RAW_DIR / "shopping_click" / "absolute.csv")
    print(f"  search_absolute : {search_df.shape[0]}개 키워드 × {search_df.shape[1]}일")
    print(f"  click_absolute  : {click_df.shape[0]}개 키워드 × {click_df.shape[1]}일")

    # ── 공통 키워드 / 날짜 ──
    common_kw    = search_df.index.intersection(click_df.index)
    all_dates    = search_df.columns.intersection(click_df.columns).sort_values()
    print(f"\n  공통 키워드: {len(common_kw)}개")
    print(f"  공통 날짜  : {len(all_dates)}일  "
          f"({all_dates[0].date()} ~ {all_dates[-1].date()})")

    # 예측 시점 필터
    pred_mask  = (all_dates >= DATASET_START) & (all_dates <= DATASET_END)
    pred_dates = all_dates[pred_mask]
    pred_idxs  = np.where(pred_mask)[0]
    print(f"  예측 시점  : {len(pred_dates)}일  "
          f"({pred_dates[0].date()} ~ {pred_dates[-1].date()})")

    # ── 피처 계산 ──
    print("\n[2/4] 피처 계산 중...")
    records   = []
    n_kw      = len(common_kw)
    n_skipped = 0

    for ki, kw in enumerate(common_kw, 1):
        if ki % 100 == 0 or ki == n_kw:
            print(f"  {ki}/{n_kw} 키워드 완료  (현재 샘플 수: {len(records):,})")

        # 전체 시계열 (공통 날짜 기준)
        s_raw = search_df.loc[kw, all_dates]
        c_raw = click_df.loc[kw, all_dates]

        # 원본 유효 마스크 (fill 이전)
        valid_s = ~s_raw.isna().values
        valid_c = ~c_raw.isna().values

        # NaN fill
        s_arr = fill_nan(s_raw)
        c_arr = fill_nan(c_raw)

        for t_idx in pred_idxs:
            row = compute_row(s_arr, c_arr, t_idx, valid_s, valid_c)
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
        "search_ma_7d", "search_growth_3d", "search_growth_7d", "search_growth_14d",
        "search_slope_7d", "search_slope_14d", "search_acceleration", "search_std_7d",
        "search_pos_14d", "days_since_max_14d",
        "click_ma_7d", "click_growth_3d", "click_growth_7d", "click_growth_14d",
        "click_slope_7d", "click_slope_14d", "click_acceleration", "click_std_7d",
        "click_pos_14d", "days_since_click_max_14d",
        "click_search_ratio", "click_lead_signal_7d", "engagement_force", "click_peak_gap",
    ]
    LABEL_COLS = ["virality_score", "peak_time"]

    df = df[["keyword", "date"] + FEATURE_COLS + LABEL_COLS]
    df = df.sort_values(["keyword", "date"]).reset_index(drop=True)

    # ── 저장 ──
    print("\n[4/4] 저장 중...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "dataset_phase1.csv"
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

    # NaN 잔존 확인
    nan_counts = df[FEATURE_COLS + LABEL_COLS].isna().sum()
    if nan_counts.any():
        print(f"\n[경고] NaN 잔존:\n{nan_counts[nan_counts > 0]}")
    else:
        print("\n[OK] NaN 없음")


if __name__ == "__main__":
    main()
