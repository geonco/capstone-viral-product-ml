# ratio ↔ absolute 변환 공유 모듈
# - estimate_max_value: 최소 diff 기반 max_value 추정
# - convert_to_absolute: ratio DataFrame → 절대값 DataFrame 변환
# - roundtrip_check: absolute → ratio 재계산 후 원본 비교

import numpy as np
import pandas as pd


def estimate_max_value(ratios, tol=0.05):
    unique = np.sort(np.unique(ratios[ratios > 0]))
    if len(unique) < 2:
        return None
    diffs = np.diff(unique)
    diffs = diffs[diffs > 1e-6]
    if len(diffs) == 0:
        return None
    base = round(100 / np.min(diffs))
    candidates = set()
    for m in [1, 2, 3, 4, 5]:
        candidates.add(base * m)
        if base // m >= 1:
            candidates.add(base // m)
    for mv in sorted(candidates):
        if mv < 1:
            continue
        restored = ratios[ratios > 0] * mv / 100
        if np.all(np.abs(restored - np.round(restored)) < tol):
            return int(mv)
    return int(base)


def convert_to_absolute(df):
    rows = {}
    n_kw = len(df.index)
    n_success, n_fallback, n_fail = 0, 0, 0

    for i, keyword in enumerate(df.index, 1):
        ratios = df.loc[keyword].values.astype(float)
        valid = ratios[~np.isnan(ratios)]
        mv = estimate_max_value(valid)

        if mv is None:
            n_fail += 1
            rows[keyword] = df.loc[keyword]
            continue

        restored = valid * mv / 100
        max_error = np.max(np.abs(restored - np.round(restored)))
        if max_error < 0.05:
            n_success += 1
        else:
            n_fallback += 1

        absolute = df.loc[keyword] * mv / 100
        rows[keyword] = absolute.round().astype("Int64")

        if i % 100 == 0 or i == n_kw:
            print(f"[{i}/{n_kw}] success={n_success} fallback={n_fallback} fail={n_fail}")

    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "keyword"
    return result, n_success, n_fallback, n_fail


def roundtrip_check(trend_df, abs_df, max_vals):
    results = {}
    for kw in abs_df.index:
        mv = max_vals.get(kw)
        if not mv:
            results[kw] = (np.nan, np.nan)
            continue
        abs_s = abs_df.loc[kw].dropna().astype(float)
        orig = trend_df.loc[kw].reindex(abs_s.index).dropna().astype(float)
        common = abs_s.index.intersection(orig.index)
        if len(common) == 0:
            results[kw] = (np.nan, np.nan)
            continue
        recon = abs_s[common] / mv * 100
        errs = np.abs(recon.values - orig[common].values)
        results[kw] = (float(np.mean(errs)), float(np.max(errs)))
    return results
