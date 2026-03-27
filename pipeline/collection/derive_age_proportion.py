# 연령별 쇼핑 클릭 ratio에서 연령대 비율(%) 산출
# age_10~60.csv → age_10_pct~age_60_pct.csv
# 같은 키워드·날짜에서 각 연령대 ratio / 전체 합산 × 100 산출
# API가 전체 그룹을 한 번에 반환하므로 일부 그룹 NaN은 0 취급

from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "shopping_click"

AGE_GROUPS = [10, 20, 30, 40, 50, 60]
INPUT_PATHS = {a: DATA_DIR / f"age_{a}.csv" for a in AGE_GROUPS}
OUTPUT_PATHS = {a: DATA_DIR / f"age_{a}_pct.csv" for a in AGE_GROUPS}


def compute_proportion(dfs):
    # NaN → 0 치환 (API가 전체 그룹을 한 번에 반환하므로 NaN = 값이 너무 작아 기록되지 않음)
    filled = {a: dfs[a].fillna(0) for a in AGE_GROUPS}

    total = sum(filled[a] for a in AGE_GROUPS)

    # 전체 그룹 NaN → 비율도 NaN
    all_nan = reduce(lambda a, b: a & b, (dfs[a].isna() for a in AGE_GROUPS))

    # 합산 = 0 → 0 나누기 방지
    total_safe = total.replace(0, np.nan)

    result = {}
    for a in AGE_GROUPS:
        pct = filled[a] / total_safe * 100
        pct[all_nan] = np.nan
        pct.index.name = "keyword"
        result[a] = pct

    return result


def main():
    for a, path in INPUT_PATHS.items():
        if not path.exists():
            print(f"missing: {path.name}")
            return

    dfs = {}
    for a, path in INPUT_PATHS.items():
        dfs[a] = pd.read_csv(path, encoding="utf-8-sig", index_col="keyword")
    print(f"loaded: {dfs[10].shape[0]} keywords x {dfs[10].shape[1]} days")

    result = compute_proportion(dfs)

    # sanity check: 6개 그룹 합 == 100%
    pct_sum = sum(result[a] for a in AGE_GROUPS)
    vals = pct_sum.values[~np.isnan(pct_sum.values)]
    if len(vals) > 0:
        max_dev = np.max(np.abs(vals - 100.0))
        print(f"sanity: max deviation from 100% = {max_dev:.10f}")

    for a, path in OUTPUT_PATHS.items():
        df = result[a].reindex(sorted(result[a].columns), axis=1)
        df.to_csv(path, encoding="utf-8-sig")
        print(f"saved: {path.name} ({df.shape[0]}x{df.shape[1]})")


if __name__ == "__main__":
    main()
