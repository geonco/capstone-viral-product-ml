# 통합 검색 트렌드 ratio(0~100)에서 절대값 역산

from pathlib import Path

import pandas as pd

from ratio_to_absolute import convert_to_absolute

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TREND_PATH = PROJECT_ROOT / "data" / "raw" / "search" / "trend.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "search" / "absolute.csv"


def main():
    if not TREND_PATH.exists():
        return

    df = pd.read_csv(TREND_PATH, encoding="utf-8-sig", index_col="keyword")
    print(f"loaded: {df.shape[0]} keywords x {df.shape[1]} days")

    result, n_ok, n_fb, n_fail = convert_to_absolute(df)
    result.to_csv(OUTPUT_PATH, encoding="utf-8-sig")
    print(f"saved: {OUTPUT_PATH.name} ({n_ok} success, {n_fb} fallback, {n_fail} fail)")


if __name__ == "__main__":
    main()
