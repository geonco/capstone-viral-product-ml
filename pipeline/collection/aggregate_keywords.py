# 윈도우별 CSV 검증 + 통합 + appearance 사분위 계층 샘플링으로 키워드 선정
# crawl_keywords.py가 저장한 CSV들을 읽어서 처리

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 프로젝트 루트 및 설정 로딩
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
with open(PROJECT_ROOT / "configs" / "collection.yaml", encoding="utf-8") as f:
    _full = yaml.safe_load(f)
    _crawl_cfg = _full["crawl"]
    _agg_cfg = _full["aggregate"]

WINDOW_DIR    = PROJECT_ROOT / "data" / "raw" / "keyword_windows"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "raw"
CAT_1ST       = _crawl_cfg["cat_1st_name"]
CAT_2ND       = _crawl_cfg["cat_2nd_name"]
FINAL_N       = _agg_cfg["final_n"]
MIN_APPEARANCE = _agg_cfg["min_appearance"]
SEED          = _agg_cfg["seed"]
EXPECTED_COLS = {"rank", "keyword", "window", "window_start", "window_end"}

# 층별 배분 (appearance 사분위 기반)
QUARTILE_LABELS = ["Q1_rare", "Q2_low", "Q3_mid", "Q4_freq"]
STRATA_ALLOC = _agg_cfg["strata"]


def validate_csvs(csv_files: list[Path]) -> list[Path]:
    valid = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig", nrows=1)
        except Exception as e:
            print(f"  [불량] {f.name}: 읽기 실패 ({e})")
            continue
        if EXPECTED_COLS - set(df.columns):
            print(f"  [불량] {f.name}: 컬럼 누락")
            continue
        if len(pd.read_csv(f, encoding="utf-8-sig")) == 0:
            print(f"  [불량] {f.name}: 행 0개")
            continue
        valid.append(f)
    return valid


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    # 기본 집계
    agg = (
        df.groupby("keyword")
        .agg(
            appearance=("window", "nunique"),
            best_rank=("rank", "min"),
            mean_rank=("rank", "mean"),
            rank_std=("rank", "std"),
        )
        .reset_index()
    )
    agg["rank_std"] = agg["rank_std"].fillna(0)

    # 월간 순위 급등폭 (이전 윈도우 대비)
    windows = sorted(df["window"].unique())
    df = df.copy()
    df["window_idx"] = df["window"].map({w: i for i, w in enumerate(windows)})
    df = df.sort_values(["keyword", "window_idx"])
    df["rank_jump"] = df.groupby("keyword")["rank"].shift(1) - df["rank"]

    max_jump = df.groupby("keyword")["rank_jump"].max().reset_index()
    max_jump.columns = ["keyword", "max_rank_jump"]
    agg = agg.merge(max_jump, on="keyword", how="left")
    agg["max_rank_jump"] = agg["max_rank_jump"].fillna(0)

    # 첫/마지막 등장
    first_last = df.groupby("keyword")["window"].agg(["min", "max"]).reset_index()
    first_last.columns = ["keyword", "first_seen", "last_seen"]
    agg = agg.merge(first_last, on="keyword", how="left")

    return agg


def assign_quartile_strata(pool: pd.DataFrame) -> pd.DataFrame:
    """appearance 기반 사분위로 층 배정 (pd.qcut)."""
    pool = pool.copy()
    try:
        pool["stratum"] = pd.qcut(
            pool["appearance"], q=4, labels=QUARTILE_LABELS,
        )
    except ValueError:
        # appearance 값이 몰려서 bin edge가 겹칠 경우 rank로 풀어줌
        pool["stratum"] = pd.qcut(
            pool["appearance"].rank(method="first"), q=4, labels=QUARTILE_LABELS,
        )
    return pool


def stratified_select(pool: pd.DataFrame, alloc: dict, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []

    for stratum, n in alloc.items():
        candidates = pool[pool["stratum"] == stratum]
        actual = min(n, len(candidates))
        if actual == 0:
            print(f"  {stratum}: 후보 없음")
            continue
        chosen = candidates.sample(n=actual, random_state=rng.integers(1e9))
        parts.append(chosen)
        print(f"  {stratum}: {actual}개 선정 (후보 {len(candidates)}개)")

    return pd.concat(parts).sort_values("appearance", ascending=False).reset_index(drop=True)


def main() -> None:
    csv_files = sorted(WINDOW_DIR.glob("kw_*.csv"))
    if not csv_files:
        print(f"{WINDOW_DIR}에 CSV 파일 없음")
        return

    # 검증
    print(f"검증: {len(csv_files)}개 CSV")
    valid = validate_csvs(csv_files)
    print(f"유효: {len(valid)}개 / 전체: {len(csv_files)}개\n")
    if not valid:
        return

    # 통합
    frames = [pd.read_csv(f, encoding="utf-8-sig") for f in valid]
    df = pd.concat(frames, ignore_index=True)
    print(f"통합: {len(df)}행, 고유 키워드 {df['keyword'].nunique()}개")

    # 전체 저장
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_path = OUTPUT_DIR / f"keywords_all_{CAT_1ST}_{CAT_2ND}.csv"
    df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"전체 저장: {all_path}\n")

    # 통계 계산 + 필터
    stats = compute_stats(df)
    pool = stats[stats["appearance"] >= MIN_APPEARANCE].copy()
    print(f"선정 풀: {len(pool)}개 (appearance >= {MIN_APPEARANCE})")

    # 층 분류 (appearance 사분위)
    pool = assign_quartile_strata(pool)
    try:
        bins = pd.qcut(pool["appearance"], q=4, retbins=True)[1].tolist()
    except ValueError:
        bins = pd.qcut(pool["appearance"].rank(method="first"), q=4, retbins=True)[1].tolist()
    print(f"사분위 경계: {bins}")
    print(f"층별 분포: {pool['stratum'].value_counts().sort_index().to_dict()}\n")

    # 시드 결정
    seed = SEED if SEED is not None else int(np.random.default_rng().integers(0, 2**31))
    print(f"시드: {seed}")

    # 계층 샘플링
    print("선정:")
    selected = stratified_select(pool, STRATA_ALLOC, seed)

    # 저장 (파일명에 시드 포함)
    out_cols = ["keyword", "stratum", "appearance", "best_rank", "rank_std",
                "first_seen", "last_seen"]
    selected_path = OUTPUT_DIR / f"keywords_selected_{FINAL_N}_seed{seed}.csv"
    selected[out_cols].to_csv(selected_path, index=False, encoding="utf-8-sig")
    print(f"\n저장: {selected_path} ({len(selected)}개)")

    # 층별 요약 출력
    for s in QUARTILE_LABELS:
        group = selected[selected["stratum"] == s]
        if group.empty:
            continue
        print(f"\n--- {s} ({len(group)}개) ---")
        print(group[["keyword", "appearance", "best_rank", "rank_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
