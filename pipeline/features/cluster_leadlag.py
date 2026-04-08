"""
키워드 행동 클러스터 기반 lead-lag 교차상관 분석

각 키워드의 시계열 특성(검색량 수준, spikiness, 변동성, 소셜 커버리지)으로
클러스터링한 뒤, 클러스터별로 blog/instagram/click → search 간
lead-lag 교차상관을 계산하여 viral/spiky 키워드와 stable 키워드 간
lead-lag 패턴 차이 검증
"""

import warnings, sys
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ── 1. 데이터 로드 ──────────────────────────────────────────────

BASE = str(Path(__file__).resolve().parent.parent.parent / "data" / "raw")

# search: wide format (keyword × date)
search_wide = pd.read_csv(f"{BASE}/search/absolute.csv", encoding="utf-8-sig")
search_wide = search_wide.rename(columns={search_wide.columns[0]: "keyword"})
search_long = search_wide.melt(id_vars="keyword", var_name="date", value_name="search")
search_long["date"] = pd.to_datetime(search_long["date"])

# click: wide format
click_wide = pd.read_csv(f"{BASE}/shopping_click/absolute.csv", encoding="utf-8-sig")
click_wide = click_wide.rename(columns={click_wide.columns[0]: "keyword"})
click_long = click_wide.melt(id_vars="keyword", var_name="date", value_name="click")
click_long["date"] = pd.to_datetime(click_long["date"])

# sometrend: long format
some = pd.read_csv(f"{BASE}/sometrends/sometrend_mention_long.csv", encoding="utf-8-sig")
some.columns = [c.strip().replace("\ufeff", "") for c in some.columns]
some["date"] = pd.to_datetime(some["date"])
some = some.rename(columns={"블로그": "blog", "인스타그램": "instagram"})
some = some[["keyword", "date", "blog", "instagram"]]

# 병합
df = search_long.merge(click_long, on=["keyword", "date"], how="inner")
df = df.merge(some, on=["keyword", "date"], how="inner")

# 결측 → 0
for c in ["search", "click", "blog", "instagram"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

common_kw = df["keyword"].unique()
print(f"공통 키워드 수: {len(common_kw)}")
print(f"날짜 범위: {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"총 행 수: {len(df):,}")

# ── 2. 키워드별 요약 피처 ──────────────────────────────────────

def keyword_features(g):
    s = g["search"]
    b = g["blog"]
    i = g["instagram"]

    # 일간 수익률 (log return, 0 회피)
    s_shift = s.shift(1)
    daily_ret = np.where(
        (s > 0) & (s_shift > 0),
        np.log(s / s_shift),
        0.0,
    )

    n_days = len(s)
    return pd.Series({
        "search_mean": s.mean(),
        "search_max": s.max(),
        "search_spikiness": s.max() / s.mean() if s.mean() > 0 else 0,
        "search_ret_std": np.std(daily_ret),
        "search_max_daily_ret": np.max(np.abs(daily_ret)),
        "blog_coverage": (b > 0).sum() / n_days,
        "insta_coverage": (i > 0).sum() / n_days,
    })

feat = df.groupby("keyword").apply(keyword_features).reset_index()
print(f"\n피처 통계:\n{feat.describe().round(3)}")

# ── 3. KMeans 클러스터링 (k=4) ──────────────────────────────────

cluster_cols = [
    "search_mean", "search_spikiness", "search_ret_std",
    "search_max_daily_ret", "blog_coverage", "insta_coverage",
]

X = feat[cluster_cols].values
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

km = KMeans(n_clusters=4, random_state=42, n_init=20)
feat["cluster"] = km.fit_predict(X_sc)

# 클러스터 특성 요약으로 이름 부여
cluster_profile = feat.groupby("cluster")[cluster_cols].mean()
print("\n── 클러스터 프로파일 (평균) ──")
print(cluster_profile.round(3))

# 클러스터 이름 자동 부여 로직
# spikiness 기준 정렬
profiles = cluster_profile.to_dict(orient="index")
cluster_names = {}
for cid, p in profiles.items():
    spiky = p["search_spikiness"]
    vol = p["search_mean"]
    ret_std = p["search_ret_std"]
    social = p["blog_coverage"] + p["insta_coverage"]

    if spiky >= cluster_profile["search_spikiness"].quantile(0.75):
        if vol >= cluster_profile["search_mean"].median():
            cluster_names[cid] = "viral_high_volume"
        else:
            cluster_names[cid] = "viral_spiky"
    elif vol >= cluster_profile["search_mean"].quantile(0.75):
        cluster_names[cid] = "steady_high_volume"
    elif vol <= cluster_profile["search_mean"].quantile(0.25):
        if social <= cluster_profile[["blog_coverage", "insta_coverage"]].sum(axis=1).quantile(0.25):
            cluster_names[cid] = "dormant_niche"
        else:
            cluster_names[cid] = "stable_low_volume"
    else:
        cluster_names[cid] = "moderate_stable"

# 중복 이름 방지: 미할당 클러스터에 순차 이름
used = set()
for cid in sorted(cluster_names):
    name = cluster_names[cid]
    if name in used:
        cluster_names[cid] = name + f"_{cid}"
    used.add(cluster_names[cid])

feat["cluster_name"] = feat["cluster"].map(cluster_names)

print("\n── 클러스터별 키워드 수 ──")
print(feat["cluster_name"].value_counts())

print("\n── 클러스터별 특성 요약 ──")
summary = feat.groupby("cluster_name").agg(
    n_keywords=("keyword", "count"),
    search_mean_avg=("search_mean", "mean"),
    search_mean_med=("search_mean", "median"),
    spikiness_avg=("search_spikiness", "mean"),
    spikiness_med=("search_spikiness", "median"),
    ret_std_avg=("search_ret_std", "mean"),
    max_daily_ret_avg=("search_max_daily_ret", "mean"),
    blog_cov_avg=("blog_coverage", "mean"),
    insta_cov_avg=("insta_coverage", "mean"),
).round(3)
print(summary.to_string())

# ── 4. 클러스터별 lead-lag 교차상관 분석 ──────────────────────

def xcorr_at_lags(x, y, lags):
    """x, y 시계열 간 lag별 Pearson 상관. lag>0 → x가 y보다 lag일 앞섬."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    # z-score
    xm = x - np.nanmean(x)
    ym = y - np.nanmean(y)
    sx = np.nanstd(x)
    sy = np.nanstd(y)
    if sx == 0 or sy == 0:
        return {lag: 0.0 for lag in lags}
    n = len(x)
    result = {}
    for lag in lags:
        if lag >= 0:
            a = xm[:n - lag] if lag > 0 else xm
            b = ym[lag:]
        else:
            a = xm[-lag:]
            b = ym[:n + lag]
        if len(a) < 15:
            result[lag] = 0.0
            continue
        r = np.nansum(a * b) / (len(a) * sx * sy)
        result[lag] = r
    return result


LAGS = list(range(-14, 15))
PAIRS = [
    ("blog", "search", "blog → search"),
    ("instagram", "search", "instagram → search"),
    ("click", "search", "click → search"),
]

results = []

for cname in sorted(feat["cluster_name"].unique()):
    kws = feat.loc[feat["cluster_name"] == cname, "keyword"].values
    cluster_df = df[df["keyword"].isin(kws)].copy()

    for src_col, tgt_col, pair_label in PAIRS:
        optimal_lags = []
        lead_counts = 0
        total_valid = 0
        all_corrs = {lag: [] for lag in LAGS}

        for kw in kws:
            kw_data = cluster_df[cluster_df["keyword"] == kw].sort_values("date")
            if len(kw_data) < 30:
                continue
            x = kw_data[src_col].values
            y = kw_data[tgt_col].values

            corrs = xcorr_at_lags(x, y, LAGS)
            for lag in LAGS:
                all_corrs[lag].append(corrs[lag])

            best_lag = max(corrs, key=corrs.get)
            optimal_lags.append(best_lag)
            if best_lag > 0:
                lead_counts += 1
            total_valid += 1

        if total_valid == 0:
            continue

        mean_corr_by_lag = {lag: np.mean(vals) for lag, vals in all_corrs.items()}
        overall_best_lag = max(mean_corr_by_lag, key=mean_corr_by_lag.get)
        overall_best_corr = mean_corr_by_lag[overall_best_lag]

        results.append({
            "cluster": cname,
            "pair": pair_label,
            "n_keywords": total_valid,
            "mean_optimal_lag": np.mean(optimal_lags),
            "median_optimal_lag": np.median(optimal_lags),
            "pct_first_leads": lead_counts / total_valid * 100,
            "cluster_agg_best_lag": overall_best_lag,
            "cluster_agg_best_corr": overall_best_corr,
            "corr_at_lag0": mean_corr_by_lag.get(0, 0),
        })

res_df = pd.DataFrame(results)

print("\n" + "=" * 100)
print("── 클러스터별 Lead-Lag 분석 결과 ──")
print("=" * 100)

# lag 부호 해석: lag > 0 → 첫 번째 신호(blog/insta/click)가 search를 lead
# lag < 0 → search가 첫 번째 신호를 lead (첫 신호가 lag)
print("\nlag > 0 : 첫 번째 신호가 search를 선행 (lead)")
print("lag < 0 : search가 첫 번째 신호를 선행 (첫 신호가 후행)")
print()

for cname in sorted(feat["cluster_name"].unique()):
    cdata = res_df[res_df["cluster"] == cname]
    n_kw = cdata["n_keywords"].iloc[0] if len(cdata) > 0 else 0
    cprofile = feat[feat["cluster_name"] == cname]
    print(f"\n┌─ {cname} (n={n_kw}) ─────────────────────────────────────")
    print(f"│  search_mean={cprofile['search_mean'].mean():.1f}  "
          f"spikiness={cprofile['search_spikiness'].mean():.1f}  "
          f"ret_std={cprofile['search_ret_std'].mean():.3f}  "
          f"blog_cov={cprofile['blog_coverage'].mean():.2f}  "
          f"insta_cov={cprofile['insta_coverage'].mean():.2f}")

    for _, row in cdata.iterrows():
        print(f"│  {row['pair']:20s} | "
              f"mean_opt_lag={row['mean_optimal_lag']:+5.1f}  "
              f"median_opt_lag={row['median_optimal_lag']:+5.1f}  "
              f"%lead={row['pct_first_leads']:5.1f}%  "
              f"agg_best_lag={row['cluster_agg_best_lag']:+3d}  "
              f"agg_best_corr={row['cluster_agg_best_corr']:.3f}  "
              f"corr@0={row['corr_at_lag0']:.3f}")
    print("└" + "─" * 80)

# ── 5. Viral/Spiky 클러스터 심층 분석 ──────────────────────────

print("\n" + "=" * 100)
print("── Viral/Spiky 클러스터 vs Stable 클러스터 비교 ──")
print("=" * 100)

# viral 클러스터 식별: spikiness가 가장 높은 클러스터
spikiness_by_cluster = feat.groupby("cluster_name")["search_spikiness"].mean()
viral_name = spikiness_by_cluster.idxmax()
# stable: spikiness가 가장 낮은 클러스터
stable_name = spikiness_by_cluster.idxmin()

print(f"\nViral/Spiky 클러스터: {viral_name}")
print(f"Most Stable 클러스터: {stable_name}")

for target_name, label in [(viral_name, "VIRAL/SPIKY"), (stable_name, "STABLE")]:
    print(f"\n── {label}: {target_name} ──")
    target_data = res_df[res_df["cluster"] == target_name]
    for _, row in target_data.iterrows():
        direction = "LEADS search" if row["mean_optimal_lag"] > 0 else "LAGS search"
        print(f"  {row['pair']:20s} → {direction} by {abs(row['mean_optimal_lag']):.1f} days "
              f"(median={row['median_optimal_lag']:+.0f}, %lead={row['pct_first_leads']:.0f}%, "
              f"best_corr={row['cluster_agg_best_corr']:.3f})")

# 비교 요약표
print("\n── Viral vs Stable 핵심 비교 ──")
print(f"{'Pair':22s} | {'Viral opt_lag':>14s} | {'Stable opt_lag':>14s} | {'Viral %lead':>12s} | {'Stable %lead':>12s}")
print("-" * 85)
for pair_label in ["blog → search", "instagram → search", "click → search"]:
    v = res_df[(res_df["cluster"] == viral_name) & (res_df["pair"] == pair_label)]
    s = res_df[(res_df["cluster"] == stable_name) & (res_df["pair"] == pair_label)]
    if len(v) > 0 and len(s) > 0:
        print(f"{pair_label:22s} | {v.iloc[0]['mean_optimal_lag']:+14.1f} | {s.iloc[0]['mean_optimal_lag']:+14.1f} | "
              f"{v.iloc[0]['pct_first_leads']:11.1f}% | {s.iloc[0]['pct_first_leads']:11.1f}%")

# 예시 키워드
print(f"\n── Viral 클러스터 키워드 예시 (상위 10, spikiness 기준) ──")
viral_kws = feat[feat["cluster_name"] == viral_name].nlargest(10, "search_spikiness")
for _, row in viral_kws.iterrows():
    print(f"  {row['keyword']:20s}  search_mean={row['search_mean']:8.1f}  "
          f"spikiness={row['search_spikiness']:6.1f}  max_ret={row['search_max_daily_ret']:.2f}")

print(f"\n── Stable 클러스터 키워드 예시 (상위 10, spikiness 기준 낮은순) ──")
stable_kws = feat[feat["cluster_name"] == stable_name].nsmallest(10, "search_spikiness")
for _, row in stable_kws.iterrows():
    print(f"  {row['keyword']:20s}  search_mean={row['search_mean']:8.1f}  "
          f"spikiness={row['search_spikiness']:6.1f}  max_ret={row['search_max_daily_ret']:.2f}")

# ── 5b. 보조 분석: k=5 시도 ──────────────────────────────────

print("\n\n" + "=" * 100)
print("── 보조: k=5 클러스터링 ──")
print("=" * 100)

km5 = KMeans(n_clusters=5, random_state=42, n_init=20)
feat["cluster5"] = km5.fit_predict(X_sc)
profile5 = feat.groupby("cluster5")[cluster_cols].mean()
print("\nk=5 클러스터 프로파일:")
print(profile5.round(3))
print("\nk=5 클러스터 크기:")
print(feat["cluster5"].value_counts().sort_index())

# 실루엣 점수
from sklearn.metrics import silhouette_score
sil4 = silhouette_score(X_sc, feat["cluster"])
sil5 = silhouette_score(X_sc, feat["cluster5"])
print(f"\n실루엣 점수: k=4 → {sil4:.3f}, k=5 → {sil5:.3f}")
