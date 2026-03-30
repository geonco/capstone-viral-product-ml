# 성별 쇼핑 클릭 ratio에서 남/여 비율(%) 산출
# gender_m.csv + gender_f.csv → gender_m_pct.csv + gender_f_pct.csv
# 같은 키워드·날짜에서 m / (m + f) × 100, f / (m + f) × 100 산출
# API가 전체 그룹을 한 번에 반환하므로 일부 그룹 NaN은 0 취급

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "shopping_click"

GROUPS = ["m", "f"]
INPUT_PATHS = {g: DATA_DIR / f"gender_{g}.csv" for g in GROUPS}
OUTPUT_PATHS = {g: DATA_DIR / f"gender_{g}_pct.csv" for g in GROUPS}


def compute_proportion(dfs):
    # NaN → 0 치환 (API가 전체 그룹을 한 번에 반환하므로 NaN = 값이 너무 작아 기록되지 않음)
    filled = {g: dfs[g].fillna(0) for g in GROUPS}

    total = filled["m"] + filled["f"]

    # 전체 그룹 NaN → 비율도 NaN
    all_nan = dfs["m"].isna() & dfs["f"].isna()

    # 합산 = 0 → 0 나누기 방지
    total_safe = total.replace(0, np.nan)

    result = {}
    for g in GROUPS:
        pct = filled[g] / total_safe * 100
        pct[all_nan] = np.nan
        pct.index.name = "keyword"
        result[g] = pct

    return result


def main():
    for g, path in INPUT_PATHS.items():
        if not path.exists():
            print(f"missing: {path.name}")
            return

    dfs = {}
    for g, path in INPUT_PATHS.items():
        dfs[g] = pd.read_csv(path, encoding="utf-8-sig", index_col="keyword")
    print(f"loaded: {dfs['m'].shape[0]} keywords x {dfs['m'].shape[1]} days")

    result = compute_proportion(dfs)

    # sanity check: m_pct + f_pct == 100%
    pct_sum = result["m"] + result["f"]
    vals = pct_sum.values[~np.isnan(pct_sum.values)]
    if len(vals) > 0:
        max_dev = np.max(np.abs(vals - 100.0))
        print(f"sanity: max deviation from 100% = {max_dev:.10f}")

    for g, path in OUTPUT_PATHS.items():
        df = result[g].reindex(sorted(result[g].columns), axis=1)
        df.to_csv(path, encoding="utf-8-sig")
        print(f"saved: {path.name} ({df.shape[0]}x{df.shape[1]})")


if __name__ == "__main__":
    main()
