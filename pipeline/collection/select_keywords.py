# 통합 키워드 CSV에서 appearance 사분위 계층 샘플링으로 키워드 선정
# crawl_keywords.py가 생성한 keywords_all_*.csv를 입력으로 사용

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
with open(PROJECT_ROOT / "configs" / "collection.yaml", encoding="utf-8") as f:
    _full = yaml.safe_load(f)
    _crawl_cfg = _full["crawl"]
    _sel_cfg = _full["select"]

CAT_1ST       = _crawl_cfg["cat_1st_name"]
CAT_2ND       = _crawl_cfg["cat_2nd_name"]
WINDOW_DIR    = PROJECT_ROOT / "data" / "raw" / "keyword_pool" / "windows"
OUTPUT_DIR    = PROJECT_ROOT / "data" / "raw" / "keyword_select"
FINAL_N       = _sel_cfg["final_n"]
MIN_APPEARANCE = _sel_cfg["min_appearance"]
SEED          = _sel_cfg["seed"]

# 층별 배분 (appearance 사분위 기반)
QUARTILE_LABELS = ["Q1_rare", "Q2_low", "Q3_mid", "Q4_freq"]
STRATA_ALLOC = _sel_cfg["strata"]


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


# appearance 기반 사분위로 층 배정
def assign_quartile_strata(pool: pd.DataFrame) -> pd.DataFrame:
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
            continue
        chosen = candidates.sample(n=actual, random_state=rng.integers(1e9))
        parts.append(chosen)
        print(f"  {stratum}: {actual}/{len(candidates)}")

    return pd.concat(parts).sort_values("appearance", ascending=False).reset_index(drop=True)


def main() -> None:
    # 통합 CSV 로딩
    all_path = WINDOW_DIR.parent / f"keywords_all_{CAT_1ST}_{CAT_2ND}.csv"
    if not all_path.exists():
        raise FileNotFoundError(f"{all_path} not found — run crawl_keywords.py first")
    df = pd.read_csv(all_path, encoding="utf-8-sig")

    # 통계 계산 + 필터
    stats = compute_stats(df)
    pool = stats[stats["appearance"] >= MIN_APPEARANCE].copy()
    pool = assign_quartile_strata(pool)

    # 시드 결정
    seed = SEED if SEED is not None else int(np.random.default_rng().integers(0, 2**31))

    # 계층 샘플링
    print(f"select_keywords: pool={len(pool)}, seed={seed}")
    selected = stratified_select(pool, STRATA_ALLOC, seed)

    # 저장
    out_cols = ["keyword", "stratum", "appearance", "best_rank", "rank_std",
                "first_seen", "last_seen"]
    selected_path = OUTPUT_DIR / f"keywords_selected_{FINAL_N}_seed{seed}.csv"
    selected[out_cols].to_csv(selected_path, index=False, encoding="utf-8-sig")
    print(f"saved: {selected_path.name} ({len(selected)} keywords)")


if __name__ == "__main__":
    main()
