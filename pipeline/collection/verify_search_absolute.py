# 검색 트렌드 역산 절대값 교차 검증
# 검증 1: round-trip — absolute → ratio 재계산 후 원본 비교
# 검증 2: 교차 비교 — 5개 키워드 일괄 API 조회, 날짜별 전체 비교 (2라운드)
# 검증 2b: 앵커 이중 체크 — 앵커 기준 overall_max 역산으로 각 키워드 절대값 재추정
# 산출물: outputs/metrics/verify_search.csv, outputs/logs/verify_search_*.jsonl

import json
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

from ratio_to_absolute import estimate_max_value, roundtrip_check

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# .env 로딩
_env = PROJECT_ROOT / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

with open(PROJECT_ROOT / "configs" / "collection.yaml", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)["search_trend"]

HEADERS = {
    "X-Naver-Client-Id": os.environ["NAVER_CLIENT_ID"],
    "X-Naver-Client-Secret": os.environ["NAVER_CLIENT_SECRET"],
    "Content-Type": "application/json",
}

TREND_PATH = PROJECT_ROOT / "data" / "raw" / "search" / "trend.csv"
ABS_PATH = PROJECT_ROOT / "data" / "raw" / "search" / "absolute.csv"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
LOGS_DIR = PROJECT_ROOT / "outputs" / "logs"
BATCH_SIZE = 5
SEED = 42


# ── API 일괄 조회 + 로그 ──

def fetch_batch(keywords):
    body = {
        "startDate": _cfg["start_date"], "endDate": _cfg["end_date"],
        "timeUnit": "date",
        "keywordGroups": [{"groupName": kw, "keywords": [kw]} for kw in keywords],
    }

    log = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "keywords": keywords,
        "request_body": body,
        "http_status": None,
        "response_keywords": None,
        "response_days": None,
        "error": None,
    }

    for attempt in range(1, 4):
        try:
            resp = requests.post(_cfg["api_url"], headers=HEADERS, json=body)
        except Exception as e:
            log["error"] = str(e)
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
            return None, log

        log["http_status"] = resp.status_code
        if resp.status_code == 200:
            result = {}
            for r in resp.json().get("results", []):
                series = {d["period"]: d["ratio"] for d in r.get("data", [])}
                result[r["title"]] = series
            log["response_keywords"] = list(result.keys())
            log["response_days"] = [len(v) for v in result.values()]
            return result, log

        log["error"] = f"HTTP {resp.status_code}: {resp.text[:300]}"
        if resp.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt)
            continue
        return None, log

    return None, log


# ── 교차 비교: 배치 내 날짜별 비교 ──

def cross_check(batch, cross_data, abs_df, max_vals):
    overall_max = max((max_vals[kw] for kw in batch if max_vals.get(kw)), default=0)
    if overall_max == 0:
        return {}
    results = {}
    for kw in batch:
        if not max_vals.get(kw) or kw not in cross_data:
            continue
        abs_s = abs_df.loc[kw].dropna().astype(float)
        errs = []
        for date, actual in cross_data[kw].items():
            if date in abs_s.index:
                expected = float(abs_s[date]) / overall_max * 100
                errs.append(abs(expected - actual))
        if errs:
            results[kw] = {"mean_err": float(np.mean(errs)), "max_err": float(np.max(errs))}
    return results


# ── 앵커 이중 체크 ──

def anchor_check(batch, cross_data, trend_dict, max_vals):
    scored = [(kw, len(cross_data.get(kw, {})))
              for kw in batch if max_vals.get(kw) and kw in cross_data]
    scored.sort(key=lambda x: x[1], reverse=True)
    anchors = [kw for kw, _ in scored[:3]]
    if not anchors:
        return {}

    factors = []
    for a in anchors:
        mv = max_vals[a]
        indiv = trend_dict.get(a, {})
        cross = cross_data.get(a, {})
        for date, c_val in cross.items():
            i_val = indiv.get(date, 0)
            if c_val > 0.1 and i_val > 0.1:
                factors.append(mv * i_val / c_val)
    if not factors:
        return {}

    actual_overall_max = np.median(factors)

    results = {}
    for kw in batch:
        mv = max_vals.get(kw)
        if not mv or kw not in cross_data:
            continue
        vals = np.array(list(cross_data[kw].values()))
        pos = vals[vals > 0]
        if len(pos) > 0:
            estimated_max = np.max(pos) * actual_overall_max / 100
            results[kw] = round(estimated_max / mv, 4)
    return results


def main():
    if not TREND_PATH.exists() or not ABS_PATH.exists():
        print("[skip] search: data not found")
        return

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    trend_df = pd.read_csv(TREND_PATH, encoding="utf-8-sig", index_col="keyword")
    abs_df = pd.read_csv(ABS_PATH, encoding="utf-8-sig", index_col="keyword")
    keywords = list(abs_df.index)

    # 역산 max 재계산
    max_vals = {}
    for kw in keywords:
        ratios = trend_df.loc[kw].values.astype(float)
        max_vals[kw] = estimate_max_value(ratios[~np.isnan(ratios)])
    valid_kws = [kw for kw in keywords if max_vals[kw]]
    print(f"keywords={len(keywords)}, valid_abs={len(valid_kws)}")

    # [1/3] round-trip
    print("\n[1/3] roundtrip")
    rt = roundtrip_check(trend_df, abs_df, max_vals)

    # [2/3] [3/3] 교차 비교 2라운드
    trend_dict = {kw: trend_df.loc[kw].dropna().to_dict() for kw in valid_kws}

    rng = random.Random(SEED)
    rounds = []
    for _ in range(2):
        shuffled = valid_kws.copy()
        rng.shuffle(shuffled)
        batches = [shuffled[i:i + BATCH_SIZE]
                   for i in range(0, len(shuffled), BATCH_SIZE)
                   if len(shuffled[i:i + BATCH_SIZE]) >= 2]
        rounds.append(batches)

    cross_results = {kw: [None, None] for kw in keywords}
    anchor_results = {kw: [None, None] for kw in keywords}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"verify_search_{ts}.jsonl"

    with open(log_path, "w", encoding="utf-8") as log_file:
        for r_idx, batches in enumerate(rounds):
            n = len(batches)
            print(f"\n[{r_idx + 2}/3] round {r_idx + 1}: {n} batches")

            for bi, batch in enumerate(batches, 1):
                cross_data, log_entry = fetch_batch(batch)
                log_entry["round"] = r_idx + 1
                log_entry["batch_idx"] = bi
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                log_file.flush()

                if not cross_data:
                    print(f"  [{bi}/{n}] FAIL")
                    continue

                for kw, vals in cross_check(batch, cross_data, abs_df, max_vals).items():
                    cross_results[kw][r_idx] = vals

                for kw, ratio in anchor_check(batch, cross_data, trend_dict, max_vals).items():
                    anchor_results[kw][r_idx] = ratio

                if bi % 25 == 0 or bi == n:
                    print(f"  [{bi}/{n}]")

                time.sleep(_cfg.get("sleep_seconds", 0.5))

    # CSV 생성
    rows = []
    for kw in keywords:
        mv = max_vals.get(kw)
        rt_mean, rt_max = rt.get(kw, (np.nan, np.nan))
        v1, v2 = cross_results[kw]
        a1, a2 = anchor_results[kw]

        rows.append({
            "keyword": kw,
            "max_abs": mv,
            "rt_mean_err": round(rt_mean, 6) if not np.isnan(rt_mean) else None,
            "rt_max_err": round(rt_max, 6) if not np.isnan(rt_max) else None,
            "v1_mean_err": round(v1["mean_err"], 6) if v1 else None,
            "v1_max_err": round(v1["max_err"], 6) if v1 else None,
            "v1_anchor": a1,
            "v2_mean_err": round(v2["mean_err"], 6) if v2 else None,
            "v2_max_err": round(v2["max_err"], 6) if v2 else None,
            "v2_anchor": a2,
        })

    df = pd.DataFrame(rows)
    out_path = METRICS_DIR / "verify_search.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nsaved: {out_path.relative_to(PROJECT_ROOT)}")

    # 요약
    valid = df[df["max_abs"].notna()]
    both = valid[valid["v1_mean_err"].notna() & valid["v2_mean_err"].notna()]
    if len(both) > 0:
        print(f"cross mean_err avg: r1={both['v1_mean_err'].mean():.4f}, r2={both['v2_mean_err'].mean():.4f}")
        print(f"cross max_err peak: r1={both['v1_max_err'].max():.4f}, r2={both['v2_max_err'].max():.4f}")

    ab = valid[valid["v1_anchor"].notna() & valid["v2_anchor"].notna()]
    if len(ab) > 0:
        ok1 = (ab["v1_anchor"] - 1.0).abs() < 0.05
        ok2 = (ab["v2_anchor"] - 1.0).abs() < 0.05
        print(f"anchor pass (0.95~1.05): {(ok1 & ok2).sum()}/{len(ab)}")

    rt_v = valid[valid["rt_max_err"].notna()]
    if len(rt_v) > 0:
        print(f"roundtrip max_err: {rt_v['rt_max_err'].max():.6f}")

    print(f"\nlog: {log_path.relative_to(PROJECT_ROOT)}")



if __name__ == "__main__":
    main()
