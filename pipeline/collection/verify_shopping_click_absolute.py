# 쇼핑 클릭 역산 절대값 검증
# round-trip만 수행 (쇼핑인사이트 API는 복수 키워드 일괄 조회 미지원)
# 산출물: outputs/metrics/verify_shopping_click.csv

from pathlib import Path

import numpy as np
import pandas as pd

from ratio_to_absolute import estimate_max_value, roundtrip_check

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TREND_PATH = PROJECT_ROOT / "data" / "raw" / "shopping_click" / "trend.csv"
ABS_PATH = PROJECT_ROOT / "data" / "raw" / "shopping_click" / "absolute.csv"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


def main():
    if not TREND_PATH.exists() or not ABS_PATH.exists():
        return

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    trend_df = pd.read_csv(TREND_PATH, encoding="utf-8-sig", index_col="keyword")
    abs_df = pd.read_csv(ABS_PATH, encoding="utf-8-sig", index_col="keyword")
    keywords = list(abs_df.index)

    max_vals = {}
    for kw in keywords:
        ratios = trend_df.loc[kw].values.astype(float)
        max_vals[kw] = estimate_max_value(ratios[~np.isnan(ratios)])
    valid_kws = [kw for kw in keywords if max_vals[kw]]
    print(f"keywords={len(keywords)}, valid_abs={len(valid_kws)}")

    rt = roundtrip_check(trend_df, abs_df, max_vals)

    rows = []
    for kw in keywords:
        mv = max_vals.get(kw)
        rt_mean, rt_max = rt.get(kw, (np.nan, np.nan))
        rows.append({
            "keyword": kw,
            "max_abs": mv,
            "rt_mean_err": round(rt_mean, 6) if not np.isnan(rt_mean) else None,
            "rt_max_err": round(rt_max, 6) if not np.isnan(rt_max) else None,
        })

    df = pd.DataFrame(rows)
    out_path = METRICS_DIR / "verify_shopping_click.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"saved: {out_path.relative_to(PROJECT_ROOT)}")

    valid = df[df["max_abs"].notna()]
    rt_v = valid[valid["rt_max_err"].notna()]
    if len(rt_v) > 0:
        print(f"roundtrip max_err: {rt_v['rt_max_err'].max():.6f}")


if __name__ == "__main__":
    main()
