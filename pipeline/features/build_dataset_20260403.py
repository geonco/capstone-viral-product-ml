import numpy as np
import pandas as pd
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
def _out_path():
    from datetime import date
    base = ROOT / "data" / "processed"
    name = f"dataset_{date.today().strftime('%Y%m%d')}"
    path = base / f"{name}.csv"
    n = 1
    while path.exists():
        path = base / f"{name}_{n}.csv"
        n += 1
    return path

PATHS = {
    "search": RAW / "search" / "absolute.csv",
    "click": RAW / "shopping_click" / "absolute.csv",
    "mention": RAW / "sometrends" / "sometrend_mention_long.csv",
}

START = pd.Timestamp("2022-05-01")
END = pd.Timestamp("2026-02-14")
LB, FW = 60, 14
SAMPLE_STRIDE = 7  # 기본 7일 간격 (겹침 50%), 14면 겹침 0%
EPS = 1e-6
CLIP = {"level": (0, 30), "growth": (-2, 20), "cv": (0, 10), "ratio": (0, 50)}

W_FULL = [3, 5, 7, 14, 30]          # scale, level, growth, spikiness, lag
W_STAT = [7, 14, 30]                 # skew, kurt, cv, rsi (최소 7개 데이터포인트 필요)
MENTION_COLS = {"블로그": "blog", "인스타그램": "instagram"}
SIGNALS = ["search", "click", "blog", "instagram"]


# ── loading ───────────────────────────────────────────────────────────────────

def load_wide(path):
    df = pd.read_csv(path, encoding="utf-8-sig", index_col="keyword")
    df.columns = pd.to_datetime(df.columns)
    return df


def load_all():
    search = load_wide(PATHS["search"])
    click = load_wide(PATHS["click"])

    raw = pd.read_csv(PATHS["mention"], encoding="utf-8-sig")
    mention = {}
    for col_kr, prefix in MENTION_COLS.items():
        p = raw.pivot(index="keyword", columns="date", values=col_kr)
        p.columns = pd.to_datetime(p.columns)
        mention[prefix] = p

    return search, click, mention


# ── nan fill ──────────────────────────────────────────────────────────────────

def fill_sig(s):
    return s.ffill().fillna(0.0).values.astype(np.float64)


def _clip(val, kind):
    lo, hi = CLIP[kind]
    return float(np.clip(val, lo, hi))


# ── RSI ───────────────────────────────────────────────────────────────────────

def _rsi(arr, period=14):
    if len(arr) < period + 1:
        return 50.0
    diffs = np.diff(arr[-(period + 1):])
    gains = np.where(diffs > 0, diffs, 0)
    losses = np.where(diffs < 0, -diffs, 0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss < EPS:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


# ── signal features (39 per signal) ──────────────────────────────────────────

def sig_feat(lb, p):
    # lb: 60-day lookback, p: prefix
    baseline = float(np.mean(lb))
    f = {}

    # A. Scale — 절대 규모 (5)
    for n in W_FULL:
        f[f"{p}_volume_{n}d"] = float(np.mean(lb[-n:]))

    # B. Position — 상대 위치 (5)
    for n in W_FULL:
        f[f"{p}_level_{n}d"] = _clip(np.mean(lb[-n:]) / (baseline + EPS), "level")

    # C. Momentum — 성장률 (5)
    for n in W_FULL:
        r, pr = lb[-n:], lb[-2 * n : -n]
        f[f"{p}_growth_{n}d"] = _clip(np.mean(r) / (np.mean(pr) + EPS) - 1, "growth")

    # D. Acceleration — 가속도 (3)
    g = lambda n: f[f"{p}_growth_{n}d"]
    f[f"{p}_accel_short"] = g(3) - g(7)
    f[f"{p}_accel_mid"] = g(7) - g(14)
    f[f"{p}_accel_long"] = g(14) - g(30)

    # E. RSI (3)
    for n in W_STAT:
        f[f"{p}_rsi_{n}d"] = _rsi(lb, period=n)

    # F. Shape — 분포 형태 (11)
    for n in W_STAT:
        w = lb[-n:]
        s = float(np.std(w))
        if s > 0:
            z = (w - np.mean(w)) / s
            f[f"{p}_skew_{n}d"] = float(np.mean(z ** 3))
            f[f"{p}_kurt_{n}d"] = float(np.mean(z ** 4) - 3.0)
        else:
            f[f"{p}_skew_{n}d"] = 0.0
            f[f"{p}_kurt_{n}d"] = 0.0
    for n in W_FULL:
        w = lb[-n:]
        f[f"{p}_spikiness_{n}d"] = float(np.max(w)) / (float(np.mean(w)) + EPS)

    # G. Stability — 안정성 (4)
    for n in W_STAT:
        w = lb[-n:]
        f[f"{p}_cv_{n}d"] = _clip(np.std(w) / (np.mean(w) + EPS), "cv")
    cv_recent = float(np.std(lb[-14:])) / (float(np.mean(lb[-14:])) + EPS)
    cv_past = float(np.std(lb[:30])) / (float(np.mean(lb[:30])) + EPS)
    f[f"{p}_cv_change"] = cv_recent - cv_past

    # H. Lag — 과거 실제값 (5)
    for n in [1, 3, 7, 14, 30]:
        f[f"{p}_lag_{n}d"] = float(lb[-n]) / (baseline + EPS)

    # I. Raw Std — 절대 변동 크기 (3)
    for n in W_STAT:
        f[f"{p}_std_{n}d"] = float(np.std(lb[-n:]))

    # J. Position — 최근 피크 대비 위치 (5)
    smooth = float(np.mean(lb[-3:]))
    for n in W_FULL:
        f[f"{p}_pos_{n}d"] = smooth / (float(np.max(lb[-n:])) + EPS)

    # K. Peak Recency — 최고점 경과일 (1)
    f[f"days_since_{p}_max_14d"] = 13 - int(np.argmax(lb[-14:]))

    return f


# ── cross features (8) ───────────────────────────────────────────────────────

def cross_feat(sf, cf, pfs):
    f = {}
    s7 = sf["search_level_7d"]

    f["click_search_ratio"] = _clip(cf["click_level_7d"] / (s7 + EPS), "ratio")
    f["click_lead_7d"] = cf["click_growth_7d"] - sf["search_growth_7d"]
    f["click_lead_14d"] = cf["click_growth_14d"] - sf["search_growth_14d"]
    f["click_search_pos_gap"] = cf["click_level_7d"] - sf["search_level_7d"]

    for pp, pf in pfs.items():
        f[f"{pp}_lead_7d"] = pf[f"{pp}_growth_7d"] - sf["search_growth_7d"]
        f[f"{pp}_search_ratio"] = _clip(pf[f"{pp}_level_7d"] / (s7 + EPS), "ratio")

    return f


# ── labels ────────────────────────────────────────────────────────────────────

def compute_labels(s_fw, c_fw, b_fw, i_fw, s_lb, c_lb, b_lb, i_lb, kw_stats=None):
    # A. buzz_composite — 4채널 z-score 가중 합산 (buzz_rank는 main에서 후처리)
    sigs_fw = {"search": s_fw, "click": c_fw, "blog": b_fw, "instagram": i_fw}
    sigs_lb = {"search": s_lb, "click": c_lb, "blog": b_lb, "instagram": i_lb}
    weights = {"search": 0.4, "click": 0.3, "blog": 0.2, "instagram": 0.1}

    composite = 0.0
    for sig, w in weights.items():
        mu = float(np.mean(sigs_lb[sig]))
        sigma = float(np.std(sigs_lb[sig]))
        fw_mean = float(np.mean(sigs_fw[sig]))
        z = (fw_mean - mu) / (sigma + EPS) if sigma > EPS else 0.0
        composite += w * z

    # B. momentum_score — 4채널 가중 가속도 (-1~1)
    accel_sum, w_sum = 0.0, 0.0
    for sig, w in weights.items():
        w1 = float(np.mean(sigs_fw[sig][:7]))
        w2 = float(np.mean(sigs_fw[sig][7:]))
        denom = max(w1, w2)
        if denom > EPS:
            accel = (w2 - w1) / (denom + EPS)
            accel_sum += w * accel
            w_sum += w
    momentum = accel_sum / (w_sum + EPS) if w_sum > 0 else 0.0
    momentum = float(np.clip(momentum, -1, 1))

    # C. channel_breadth — 활성 채널 수 (0~4)
    breadth = 0
    for sig in ["search", "click", "blog", "instagram"]:
        baseline = float(np.median(sigs_lb[sig]))
        if float(np.mean(sigs_fw[sig])) > baseline and baseline > EPS:
            breadth += 1

    # D. conversion_shift — 구매 전환율 변화 (z-score)
    awareness_fw = float(np.mean(s_fw)) + float(np.mean(b_fw)) + float(np.mean(i_fw))
    action_fw = float(np.mean(c_fw))
    ratio_fw = action_fw / (awareness_fw + EPS) if awareness_fw > EPS else 0.0

    awareness_lb = float(np.mean(s_lb)) + float(np.mean(b_lb)) + float(np.mean(i_lb))
    action_lb = float(np.mean(c_lb))
    ratio_lb = action_lb / (awareness_lb + EPS) if awareness_lb > EPS else 0.0

    # lookback 내 ratio의 변동으로 z-score 근사
    ratio_diff = ratio_fw - ratio_lb
    conv_shift = ratio_diff / (abs(ratio_lb) + EPS) if abs(ratio_lb) > EPS else 0.0
    conv_shift = float(np.clip(conv_shift, -5, 5))

    # ── 기존 라벨 (v1) ──

    # future_intensity — 4채널 합산 버즈 볼륨
    intensity = float(np.mean(s_fw) + np.mean(c_fw) + np.mean(b_fw) + np.mean(i_fw))

    # future_acceleration — 검색 week2/week1 성장률
    fa_w1 = float(np.mean(s_fw[:7]))
    fa_w2 = float(np.mean(s_fw[7:]))
    fa_accel = (fa_w2 - fa_w1) / (fa_w1 + EPS)
    fa_accel = float(np.clip(fa_accel, -1, 10))

    # signal_agreement — 비영·비상수 시그널 간 pairwise 상관계수 평균
    sigs_list = [arr for arr in [s_fw, c_fw, b_fw, i_fw]
                 if float(np.mean(arr)) > EPS and float(np.std(arr)) > EPS]
    if len(sigs_list) >= 2:
        corr_mat = np.corrcoef(sigs_list)
        n_sigs = len(sigs_list)
        pairs = [corr_mat[i, j] for i in range(n_sigs) for j in range(i + 1, n_sigs)]
        agreement = float(np.nanmean(pairs))
        if np.isnan(agreement):
            agreement = 0.0
    else:
        agreement = 0.0

    # search_click_convergence — 검색↔클릭 min-max 정규화 후 상관계수
    s_range = float(np.max(s_fw) - np.min(s_fw))
    c_range = float(np.max(c_fw) - np.min(c_fw))
    if s_range > EPS and c_range > EPS:
        s_norm = (s_fw - np.min(s_fw)) / (s_range + EPS)
        c_norm = (c_fw - np.min(c_fw)) / (c_range + EPS)
        conv = float(np.corrcoef(s_norm, c_norm)[0, 1])
        if np.isnan(conv):
            conv = 0.0
    else:
        conv = 0.0

    return {
        # v1 라벨
        "future_intensity": round(intensity, 6),
        "future_acceleration": round(fa_accel, 6),
        "signal_agreement": round(agreement, 6),
        "search_click_convergence": round(conv, 6),
        # v2 라벨
        "buzz_composite": round(composite, 6),
        "momentum_score": round(momentum, 6),
        "channel_breadth": breadth,
        "conversion_shift": round(conv_shift, 6),
    }


# ── single row ────────────────────────────────────────────────────────────────

def compute(s, c, plats, t, s_inv):
    lo, hi = t - LB, t + FW

    if lo < 0 or hi > len(s):
        return None
    if s_inv[lo:lo + 30].all():
        return None

    s_lb, s_fw = s[lo:t], s[t:hi]
    c_lb, c_fw = c[lo:t], c[t:hi]
    if float(np.std(s_lb)) == 0:
        return None

    sf = sig_feat(s_lb, "search")
    cf = sig_feat(c_lb, "click")
    pfs = {p: sig_feat(a[lo:t], p) for p, a in plats.items()}

    row = {}
    row.update(sf)
    row.update(cf)
    for pf in pfs.values():
        row.update(pf)
    row.update(cross_feat(sf, cf, pfs))

    # surge 피처 — blog/instagram 선행 신호 (6개)
    s_base_30 = float(np.mean(s_lb[:30]))
    for pp in ["blog", "instagram"]:
        if pp not in plats:
            row[f"{pp}_surge_7d"] = 0
            row[f"{pp}_surge_days_ago"] = -1
            row[f"{pp}_surge_before_search"] = 0
            continue
        p_lb = plats[pp][lo:t]
        p_base = float(np.mean(p_lb[:30]))

        # 최근 7일 내 baseline 2배 돌파 여부
        surge_mask = p_lb[-7:] > p_base * 2 if p_base > 0 else np.zeros(7, dtype=bool)
        row[f"{pp}_surge_7d"] = int(surge_mask.any())

        # 최근 14일 내 2배 돌파 최초 시점 (경과일)
        cross_idx = np.where(p_lb[-14:] > p_base * 2)[0] if p_base > 0 else np.array([])
        row[f"{pp}_surge_days_ago"] = int(14 - cross_idx[0]) if len(cross_idx) > 0 else -1

        # blog/insta 2배 돌파 + search 아직 안 넘음
        s_surged = (s_lb[-7:] > s_base_30 * 2).any() if s_base_30 > 0 else False
        row[f"{pp}_surge_before_search"] = int(surge_mask.any() and not s_surged)

    # calendar
    row["day_of_week"] = t  # placeholder
    row["month"] = t        # placeholder

    # 라벨 — 미래 + 과거 데이터 사용
    b_lb = plats["blog"][lo:t] if "blog" in plats else np.zeros(LB, dtype=np.float64)
    b_fw = plats["blog"][t:hi] if "blog" in plats else np.zeros(FW, dtype=np.float64)
    i_lb = plats["instagram"][lo:t] if "instagram" in plats else np.zeros(LB, dtype=np.float64)
    i_fw = plats["instagram"][t:hi] if "instagram" in plats else np.zeros(FW, dtype=np.float64)
    row.update(compute_labels(s_fw, c_fw, b_fw, i_fw, s_lb, c_lb, b_lb, i_lb))
    return row


# ── column order ──────────────────────────────────────────────────────────────

def _sig_cols(p):
    c = []
    for n in W_FULL:    c.append(f"{p}_volume_{n}d")
    for n in W_FULL:    c.append(f"{p}_level_{n}d")
    for n in W_FULL:    c.append(f"{p}_growth_{n}d")
    c += [f"{p}_accel_short", f"{p}_accel_mid", f"{p}_accel_long"]
    for n in W_STAT:    c.append(f"{p}_rsi_{n}d")
    for n in W_STAT:    c.append(f"{p}_skew_{n}d")
    for n in W_STAT:    c.append(f"{p}_kurt_{n}d")
    for n in W_FULL:    c.append(f"{p}_spikiness_{n}d")
    for n in W_STAT:    c.append(f"{p}_cv_{n}d")
    c.append(f"{p}_cv_change")
    for n in [1, 3, 7, 14, 30]: c.append(f"{p}_lag_{n}d")
    for n in W_STAT:    c.append(f"{p}_std_{n}d")
    for n in W_FULL:    c.append(f"{p}_pos_{n}d")
    c.append(f"days_since_{p}_max_14d")
    return c


FEAT_COLS = []
for _p in SIGNALS:
    FEAT_COLS += _sig_cols(_p)

FEAT_COLS += [
    "click_search_ratio", "click_lead_7d", "click_lead_14d", "click_search_pos_gap",
    "blog_lead_7d", "blog_search_ratio",
    "instagram_lead_7d", "instagram_search_ratio",
]
FEAT_COLS += [
    "blog_surge_7d", "blog_surge_days_ago", "blog_surge_before_search",
    "instagram_surge_7d", "instagram_surge_days_ago", "instagram_surge_before_search",
]
FEAT_COLS += ["day_of_week", "month"]

LABEL_COLS = [
    # v1
    "future_intensity", "future_acceleration",
    "signal_agreement", "search_click_convergence",
    # v2
    "buzz_composite", "momentum_score",
    "channel_breadth", "conversion_shift",
]
assert len(FEAT_COLS) == 216, f"expected 216, got {len(FEAT_COLS)}"


# ── main ──────────────────────────────────────────────────────────────────────

def _process_keyword(args):
    kw, s_arr, c_arr, plats, pred_idx, dates, s_inv, stride = args
    rows, skipped = [], 0
    for ti in pred_idx[::stride]:
        r = compute(s_arr, c_arr, plats, ti, s_inv)
        if r is None:
            skipped += 1
            continue
        r["keyword"] = kw
        dt = dates[ti]
        r["date"] = dt.date()
        r["day_of_week"] = dt.dayofweek
        r["month"] = dt.month
        rows.append(r)
    return rows, skipped


def main():
    import argparse
    from multiprocessing import Pool, cpu_count

    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=SAMPLE_STRIDE, help="샘플링 간격 (기본 7일)")
    args = parser.parse_args()
    stride = args.stride

    print("=" * 50)
    print(f"building dataset (216 features, stride={stride})")
    print("=" * 50)

    search, click, mention = load_all()

    kws = search.index.intersection(click.index)
    dates = search.columns.intersection(click.columns).sort_values()

    mask = (dates >= START) & (dates <= END)
    pred_idx = np.where(mask)[0]
    mention_kws = set(next(iter(mention.values())).index)

    n_workers = max(1, cpu_count() - 1)
    est_samples = len(kws) * len(pred_idx[::stride])
    print(f"  {len(kws)} keywords, {len(dates)} days, {len(pred_idx)} pred dates")
    print(f"  stride={stride} → ~{est_samples:,} samples (before filter)")
    print(f"  mention: {len(kws.intersection(mention_kws))}/{len(kws)}")
    print(f"  workers: {n_workers}")

    def gen_tasks():
        for kw in kws:
            sr = search.loc[kw, dates]
            s_inv = (sr.isna() | (sr == 0)).values
            s = fill_sig(sr)
            c = fill_sig(click.loc[kw, dates])

            plats = {}
            for pfx, mdf in mention.items():
                if kw in mdf.index:
                    plats[pfx] = fill_sig(mdf.loc[kw].reindex(dates))
                else:
                    plats[pfx] = np.zeros(len(dates), dtype=np.float64)

            yield (kw, s, c, plats, pred_idx, dates, s_inv, stride)

    chunks, total, skipped = [], 0, 0
    with Pool(n_workers, maxtasksperchild=50) as pool:
        for i, (kw_rows, kw_skipped) in enumerate(pool.imap_unordered(_process_keyword, gen_tasks(), chunksize=10), 1):
            if kw_rows:
                chunks.append(pd.DataFrame(kw_rows))
                total += len(kw_rows)
            skipped += kw_skipped
            print(f"  {i}/{len(kws)} ({total:,} samples)")

    print(f"\n  {total:,} samples ({skipped:,} skipped)")

    df = pd.concat(chunks, ignore_index=True)
    del chunks

    # buzz_rank — 날짜별 buzz_composite 백분위 (0~100)
    df["buzz_rank"] = df.groupby("date")["buzz_composite"].rank(pct=True) * 100
    df["buzz_rank"] = df["buzz_rank"].round(2)

    ALL_LABELS = LABEL_COLS + ["buzz_rank"]
    df = df[["keyword", "date"] + FEAT_COLS + ALL_LABELS]
    df.sort_values(["keyword", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    out = _out_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  saved: {out}")

    print(f"\n  shape: {df.shape}")
    for label in ALL_LABELS:
        print(f"  {label}: mean={df[label].mean():.4f}  std={df[label].std():.4f}  min={df[label].min():.2f}  max={df[label].max():.2f}")

    nans = df[FEAT_COLS + ALL_LABELS].isna().sum()
    bad = nans[nans > 0]
    print(f"\n  {'[WARN] NaN: ' + str(bad.to_dict()) if len(bad) else '[OK] no NaN'}")


if __name__ == "__main__":
    main()
