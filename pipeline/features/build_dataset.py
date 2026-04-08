import hashlib
import inspect
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
def _out_path():
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "data" / "processed" / f"dataset_{stamp}.csv"

PATHS = {
    "search": RAW / "search" / "absolute.csv",
    "click": RAW / "shopping_click" / "absolute.csv",
    "mention": RAW / "sometrends" / "sometrend_mention_long.csv",
}

START = pd.Timestamp("2022-05-01")
END = pd.Timestamp("2026-02-14")
LB, FW = 60, 14
SAMPLE_STRIDE = 1  # 빌드 시 stride=1 (전체), train/valid 서브샘플링은 train.py에서
EPS = 1e-6
CLIP = {"level": (0, 30), "growth": (-2, 20), "cv": (0, 10), "ratio": (0, 50)}

W_FULL = [3, 5, 7, 14, 30]          # 내부 계산용 (accel에 growth_3d/5d 필요)
W_OUT = [7, 14, 30]                  # 출력 윈도우 (3d/5d 제거 — SHAP 하위, 상관 >0.95)
W_STAT = [7, 14, 30]                 # skew, kurt, cv, rsi (최소 7개 데이터포인트 필요)
MENTION_COLS = {"블로그": "blog", "인스타그램": "instagram"}
SIGNALS = ["search", "click", "blog", "instagram"]


# ── cache ─────────────────────────────────────────────────────────────────────

CACHE_DIR = ROOT / "data" / "cache"
CACHE_META = CACHE_DIR / "meta.json"


def _sig_feat_hash():
    src = inspect.getsource(sig_feat)
    return hashlib.md5(src.encode()).hexdigest()


def _raw_mtime():
    return max(p.stat().st_mtime for p in PATHS.values() if p.exists())


def _load_meta():
    if CACHE_META.exists():
        return json.loads(CACHE_META.read_text())
    return {}


def _save_meta(meta):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_META.write_text(json.dumps(meta, indent=2))


def _l1_valid(meta):
    return (
        meta.get("l1_raw_mtime")
        and (CACHE_DIR / "signals.h5").exists()
        and meta["l1_raw_mtime"] >= _raw_mtime()
    )


def _l2_valid(meta):
    return (
        meta.get("l2_sig_feat_hash")
        and (CACHE_DIR / "sig_feat.parquet").exists()
        and meta["l2_sig_feat_hash"] == _sig_feat_hash()
    )


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


# ── signal features (65 per signal) ──────────────────────────────────────────

def sig_feat(lb, p):
    # lb: 60-day lookback, p: prefix
    baseline = float(np.mean(lb))
    f = {}

    # A. Scale — 절대 규모 (4: 3d 추가)
    W_SHORT = [3, 7, 14, 30]
    for n in W_SHORT:
        f[f"{p}_volume_{n}d"] = float(np.mean(lb[-n:]))

    # B. Relative Position (4: 3d 추가)
    for n in W_SHORT:
        f[f"{p}_level_{n}d"] = _clip(np.mean(lb[-n:]) / (baseline + EPS), "level")

    # C. Momentum — 성장률 (4 출력, 5 내부 계산)
    _g = {}
    for n in W_FULL:
        r, pr = lb[-n:], lb[-2 * n : -n]
        _g[n] = _clip(np.mean(r) / (np.mean(pr) + EPS) - 1, "growth")
    for n in W_SHORT:
        f[f"{p}_growth_{n}d"] = _g[n]

    # D. Acceleration (3) — 내부 growth 값 사용
    f[f"{p}_accel_short"] = _g[3] - _g[7]
    f[f"{p}_accel_mid"] = _g[7] - _g[14]
    f[f"{p}_accel_long"] = _g[14] - _g[30]

    # E. RSI (2, 7d 제거 — 7포인트 RSI = 잡음)
    for n in [14, 30]:
        f[f"{p}_rsi_{n}d"] = _rsi(lb, period=n)

    # F. Shape (skew/kurt 14d/30d + spikiness 3d 추가)
    for n in [14, 30]:
        w = lb[-n:]
        s = float(np.std(w))
        if s > 0:
            z = (w - np.mean(w)) / s
            f[f"{p}_skew_{n}d"] = float(np.mean(z ** 3))
            f[f"{p}_kurt_{n}d"] = float(np.mean(z ** 4) - 3.0)
        else:
            f[f"{p}_skew_{n}d"] = 0.0
            f[f"{p}_kurt_{n}d"] = 0.0
    for n in W_SHORT:
        w = lb[-n:]
        f[f"{p}_spikiness_{n}d"] = float(np.max(w)) / (float(np.mean(w)) + EPS)

    # G. Stability (4)
    for n in W_STAT:
        w = lb[-n:]
        f[f"{p}_cv_{n}d"] = _clip(np.std(w) / (np.mean(w) + EPS), "cv")
    cv_recent = float(np.std(lb[-14:])) / (float(np.mean(lb[-14:])) + EPS)
    cv_past = float(np.std(lb[:30])) / (float(np.mean(lb[:30])) + EPS)
    f[f"{p}_cv_change"] = cv_recent - cv_past

    # H. Lag (5)
    for n in [1, 3, 7, 14, 30]:
        f[f"{p}_lag_{n}d"] = float(lb[-n]) / (baseline + EPS)

    # I. Std (3)
    for n in W_STAT:
        f[f"{p}_std_{n}d"] = float(np.std(lb[-n:]))

    # J. Position (5)
    smooth = float(np.mean(lb[-3:]))
    for n in W_FULL:
        f[f"{p}_pos_{n}d"] = smooth / (float(np.max(lb[-n:])) + EPS)

    # K. Peak Recency (1)
    f[f"days_since_{p}_max_14d"] = 13 - int(np.argmax(lb[-14:]))

    # ── 신규 동적 피처 ──

    # L. Micro-Trend — 기울기, 곡률, 추세 품질 (8)
    slope_3d = float(np.polyfit(np.arange(3), lb[-3:], 1)[0])
    slope_7d = float(np.polyfit(np.arange(7), lb[-7:], 1)[0])
    slope_14d = float(np.polyfit(np.arange(14), lb[-14:], 1)[0])
    f[f"{p}_slope_3d"] = slope_3d
    f[f"{p}_slope_7d"] = slope_7d
    f[f"{p}_slope_14d"] = slope_14d
    f[f"{p}_norm_slope_7d"] = slope_7d / (baseline + EPS)
    f[f"{p}_slope_change"] = slope_3d - slope_7d

    std_7 = float(np.std(lb[-7:]))
    if std_7 > 0:
        r = np.corrcoef(np.arange(7), lb[-7:])[0, 1]
        r2 = r ** 2 if not np.isnan(r) else 0.0
    else:
        r2 = 0.0
    f[f"{p}_trend_r2_7d"] = r2
    f[f"{p}_residual_energy_7d"] = 1.0 - r2
    f[f"{p}_curvature_14d"] = float(np.polyfit(np.arange(14), lb[-14:], 2)[0])

    # M. Wave Dynamics — 진동, 감쇠, 방향 전환 (6)
    std_30 = float(np.std(lb[-30:]))
    f[f"{p}_vol_ratio"] = std_7 / (std_30 + EPS)

    range_7 = float(np.max(lb[-7:]) - np.min(lb[-7:]))
    range_30 = float(np.max(lb[-30:]) - np.min(lb[-30:]))
    f[f"{p}_range_squeeze"] = range_7 / (range_30 + EPS)

    amp_first = float(np.max(lb[-14:-7]) - np.min(lb[-14:-7]))
    amp_second = range_7
    f[f"{p}_damping"] = float(np.clip(amp_second / (amp_first + EPS), 0, 10))

    diffs = np.diff(lb[-14:])
    f[f"{p}_zero_crossings_14d"] = int(np.sum(diffs[:-1] * diffs[1:] < 0))

    recent_diffs = np.diff(lb[-7:])
    if len(recent_diffs) > 0 and recent_diffs[-1] != 0:
        last_sign = np.sign(recent_diffs[-1])
        streak = 1
        for i in range(len(recent_diffs) - 2, -1, -1):
            if np.sign(recent_diffs[i]) == last_sign:
                streak += 1
            else:
                break
        f[f"{p}_direction_streak"] = int(streak * last_sign)
    else:
        f[f"{p}_direction_streak"] = 0

    d3 = np.diff(lb[-4:])
    f[f"{p}_reversal_3d"] = int(d3[-1] * d3[-2] < 0) if len(d3) >= 2 else 0

    # N. Regime/Breakout — 볼린저 밴드, 돌파 강도 (4)
    mu_30 = float(np.mean(lb[-30:]))
    upper = mu_30 + 2 * std_30
    lower = mu_30 - 2 * std_30
    band_range = upper - lower
    f[f"{p}_band_position"] = float(lb[-1] - lower) / (band_range + EPS)
    f[f"{p}_band_width"] = band_range / (mu_30 + EPS)
    f[f"{p}_drawdown_14d"] = (float(lb[-1]) - float(np.max(lb[-14:]))) / (float(np.max(lb[-14:])) + EPS)
    f[f"{p}_breakout_strength"] = (float(lb[-1]) - upper) / (std_30 + EPS)

    # O. Daily Momentum — 일별 변화율, 다이버전스 (4)
    prev_val = float(lb[-2])
    f[f"{p}_daily_return_1d"] = float(np.clip((float(lb[-1]) - prev_val) / (prev_val + EPS), -10, 100))

    daily_rets = np.diff(lb[-4:]) / (np.abs(lb[-4:-1]) + EPS)
    f[f"{p}_daily_returns_3d_avg"] = float(np.clip(np.mean(daily_rets), -10, 100))

    rsi_dir = 1 if f[f"{p}_rsi_14d"] > 50 else -1
    slope_dir = 1 if slope_7d > 0 else -1
    f[f"{p}_momentum_divergence"] = int(rsi_dir != slope_dir)

    daily_accel_arr = np.diff(lb[-3:])
    f[f"{p}_daily_accel_1d"] = float(daily_accel_arr[-1] - daily_accel_arr[0]) if len(daily_accel_arr) >= 2 else 0.0

    return f


# ── cross-correlation helpers ────────────────────────────────────────────────

def _xcorr_lag(a_lb, b_lb, max_lag=7):
    # a가 b를 선행하는 최적 시차와 상관도
    a_diff = np.diff(a_lb[-15:])
    b_diff = np.diff(b_lb[-15:])
    if float(np.std(a_diff)) < EPS or float(np.std(b_diff)) < EPS:
        return 0, 0.0
    best_lag, best_corr = 0, 0.0
    for lag in range(max_lag + 1):
        if lag == 0:
            c = np.corrcoef(a_diff, b_diff)[0, 1]
        else:
            if len(a_diff) - lag < 3:
                break
            c = np.corrcoef(a_diff[:-lag], b_diff[lag:])[0, 1]
        if np.isnan(c):
            continue
        if abs(c) > abs(best_corr):
            best_lag, best_corr = lag, float(c)
    return best_lag, best_corr


def _dir_agree(a_lb, b_lb, window=7):
    a_dir = np.sign(np.diff(a_lb[-(window + 1):]))
    b_dir = np.sign(np.diff(b_lb[-(window + 1):]))
    return float(np.mean(a_dir == b_dir))


# ── cross features (20 = 8 old + 12 new) ────────────────────────────────────

def cross_feat(sf, cf, pfs, all_lbs):
    f = {}
    s7 = sf["search_level_7d"]

    # 기존 8개 — ratio, lead, pos gap
    f["click_search_ratio"] = _clip(cf["click_level_7d"] / (s7 + EPS), "ratio")
    f["click_lead_7d"] = cf["click_growth_7d"] - sf["search_growth_7d"]
    f["click_lead_14d"] = cf["click_growth_14d"] - sf["search_growth_14d"]
    f["click_search_pos_gap"] = cf["click_level_7d"] - sf["search_level_7d"]

    for pp, pf in pfs.items():
        f[f"{pp}_lead_7d"] = pf[f"{pp}_growth_7d"] - sf["search_growth_7d"]
        f[f"{pp}_search_ratio"] = _clip(pf[f"{pp}_level_7d"] / (s7 + EPS), "ratio")

    # 신규 12개 — 교차상관 시차, 방향 동조, 기울기 괴리
    s_lb = all_lbs["search"]
    for other in ["blog", "instagram", "click"]:
        prefix = "insta" if other == "instagram" else other
        o_lb = all_lbs.get(other, np.zeros_like(s_lb))
        lag, xcorr = _xcorr_lag(o_lb, s_lb)
        f[f"{prefix}_search_best_lag"] = lag
        f[f"{prefix}_search_peak_xcorr"] = xcorr
        f[f"{prefix}_search_dir_agree"] = _dir_agree(o_lb, s_lb)

    # 기울기 괴리 (blog/insta → search 선행 여부)
    f["blog_search_slope_gap"] = pfs.get("blog", {}).get("blog_norm_slope_7d", 0.0) - sf["search_norm_slope_7d"]
    f["insta_search_slope_gap"] = pfs.get("instagram", {}).get("instagram_norm_slope_7d", 0.0) - sf["search_norm_slope_7d"]

    # 다채널 모멘텀 (양의 기울기 신호 수)
    f["multi_slope_count"] = sum(
        1 for sig in ["search", "click", "blog", "instagram"]
        if all_lbs.get(sig, np.zeros(1)).shape[0] > 7
        and float(np.polyfit(np.arange(7), all_lbs[sig][-7:], 1)[0]) > 0
    )

    # ── 멀티채널 종합 피처 (4종 × 5윈도우 = 19개) ──
    weights = {"search": 0.4, "click": 0.3, "blog": 0.2, "instagram": 0.1}
    W_MULTI = [1, 3, 7, 14, 30]

    for n in W_MULTI:
        # past_conversion_rate — 과거 n일 구매 전환율
        awareness = sum(float(np.mean(all_lbs.get(s, np.zeros(1))[-n:])) for s in ["search", "blog", "instagram"])
        action = float(np.mean(all_lbs.get("click", np.zeros(1))[-n:]))
        f[f"past_conversion_rate_{n}d"] = action / (awareness + EPS) if awareness > EPS else 0.0

        # past_channel_active — 과거 n일 기준 활성 채널 수
        f[f"past_channel_active_{n}d"] = sum(
            1 for sig in ["search", "click", "blog", "instagram"]
            if all_lbs.get(sig, np.zeros(1)).shape[0] >= n
            and float(np.mean(all_lbs[sig][-n:])) > float(np.median(all_lbs[sig])) + EPS
            and float(np.median(all_lbs[sig])) > EPS
        )

        # multi_accel — 4채널 가중 가속도 (최근 n일 vs 이전 n일, n>=3만)
        if n >= 3:
            accel_sum = 0.0
            for sig, w in weights.items():
                lb = all_lbs.get(sig, np.zeros(1))
                if len(lb) >= 2 * n:
                    prev = float(np.mean(lb[-2 * n:-n]))
                    recent = float(np.mean(lb[-n:]))
                    denom = max(prev, recent)
                    if denom > EPS:
                        accel_sum += w * (recent - prev) / (denom + EPS)
            f[f"multi_accel_{n}d"] = float(np.clip(accel_sum, -1, 1))

        # past_buzz_zscore — 최근 n일 4채널 가중 z-score
        zscore_sum = 0.0
        for sig, w in weights.items():
            lb = all_lbs.get(sig, np.zeros(1))
            if len(lb) >= n:
                mu = float(np.mean(lb))
                sigma = float(np.std(lb))
                recent = float(np.mean(lb[-n:]))
                z = (recent - mu) / (sigma + EPS) if sigma > EPS else 0.0
                zscore_sum += w * z
        f[f"past_buzz_zscore_{n}d"] = zscore_sum

    return f


# ── labels ────────────────────────────────────────────────────────────────────

def compute_labels(s_fw, c_fw, b_fw, i_fw, s_lb, c_lb, b_lb, i_lb, kw_stats=None):
    # A. buzz_composite — 4채널 z-score 가중 합산
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

    # ── 순수 미래 라벨 ──

    # future_intensity — 4채널 합산 버즈 볼륨
    intensity = float(np.mean(s_fw) + np.mean(c_fw) + np.mean(b_fw) + np.mean(i_fw))

    # peak_timing — 4채널 합산 피크 시점 (0=즉시, 1=후반)
    peak_timing = int(np.argmax(s_fw + c_fw + b_fw + i_fw)) / (FW - 1)

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
        # 순수 미래
        "future_intensity": round(intensity, 6),
        "future_acceleration": round(fa_accel, 6),
        "peak_timing": round(peak_timing, 6),
        "signal_agreement": round(agreement, 6),
        "search_click_convergence": round(conv, 6),
        # 과거+미래 혼합
        "buzz_composite": round(composite, 6),
        "momentum_score": round(momentum, 6),
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
    plat_lbs = {p: a[lo:t] for p, a in plats.items()}
    pfs = {p: sig_feat(plat_lbs[p], p) for p in plat_lbs}

    row = {}
    row.update(sf)
    row.update(cf)
    for pf in pfs.values():
        row.update(pf)

    all_lbs = {"search": s_lb, "click": c_lb}
    all_lbs.update(plat_lbs)
    row.update(cross_feat(sf, cf, pfs, all_lbs))

    # surge — blog/instagram 선행 신호, 연속값 (3d/7d × 2종 + lead_days = 10개)
    for pp in ["blog", "instagram"]:
        p_lb = plat_lbs.get(pp, np.zeros(LB, dtype=np.float64))
        p_base = float(np.mean(p_lb[:30]))

        for sn in [3, 7]:
            # surge_intensity: 최근 n일 최대값 / baseline - 1
            if p_base > EPS:
                row[f"{pp}_surge_intensity_{sn}d"] = float(np.clip(np.max(p_lb[-sn:]) / p_base - 1, 0, 50))
            else:
                row[f"{pp}_surge_intensity_{sn}d"] = 0.0

            # surge_accel: 최근 n일 기울기 / baseline
            if p_base > EPS:
                row[f"{pp}_surge_accel_{sn}d"] = float(np.polyfit(np.arange(sn), p_lb[-sn:], 1)[0]) / p_base
            else:
                row[f"{pp}_surge_accel_{sn}d"] = 0.0

        # surge_lead_days: platform peak - search peak in last 14d (양수 = platform 선행)
        p_peak = int(np.argmax(p_lb[-14:]))
        s_peak = int(np.argmax(s_lb[-14:]))
        row[f"{pp}_surge_lead_days"] = s_peak - p_peak

    # calendar (day_of_week 제거 — SHAP=0)
    row["month"] = t  # placeholder, 실제값은 _process_keyword에서 덮어씀

    # 라벨 — 미래 + 과거 데이터 사용
    b_lb = plats["blog"][lo:t] if "blog" in plats else np.zeros(LB, dtype=np.float64)
    b_fw = plats["blog"][t:hi] if "blog" in plats else np.zeros(FW, dtype=np.float64)
    i_lb = plats["instagram"][lo:t] if "instagram" in plats else np.zeros(LB, dtype=np.float64)
    i_fw = plats["instagram"][t:hi] if "instagram" in plats else np.zeros(FW, dtype=np.float64)
    row.update(compute_labels(s_fw, c_fw, b_fw, i_fw, s_lb, c_lb, b_lb, i_lb))
    return row


# ── column order ──────────────────────────────────────────────────────────────

W_SHORT = [3, 7, 14, 30]

def _sig_cols(p):
    c = []
    for n in W_SHORT:   c.append(f"{p}_volume_{n}d")          # 4
    for n in W_SHORT:   c.append(f"{p}_level_{n}d")           # 4
    for n in W_SHORT:   c.append(f"{p}_growth_{n}d")          # 4
    c += [f"{p}_accel_short", f"{p}_accel_mid", f"{p}_accel_long"]  # 3
    for n in [14, 30]:  c.append(f"{p}_rsi_{n}d")             # 2
    for n in [14, 30]:  c.append(f"{p}_skew_{n}d")            # 2
    for n in [14, 30]:  c.append(f"{p}_kurt_{n}d")            # 2
    for n in W_SHORT:   c.append(f"{p}_spikiness_{n}d")       # 4
    for n in W_STAT:    c.append(f"{p}_cv_{n}d")              # 3
    c.append(f"{p}_cv_change")                                 # 1
    for n in [1, 3, 7, 14, 30]: c.append(f"{p}_lag_{n}d")    # 5
    for n in W_STAT:    c.append(f"{p}_std_{n}d")             # 3
    for n in W_FULL:    c.append(f"{p}_pos_{n}d")             # 5
    c.append(f"days_since_{p}_max_14d")                        # 1
    # L. Micro-Trend (8)
    c += [f"{p}_slope_3d", f"{p}_slope_7d", f"{p}_slope_14d"]
    c += [f"{p}_norm_slope_7d", f"{p}_slope_change"]
    c += [f"{p}_trend_r2_7d", f"{p}_residual_energy_7d", f"{p}_curvature_14d"]
    # M. Wave Dynamics (6)
    c += [f"{p}_vol_ratio", f"{p}_range_squeeze", f"{p}_damping"]
    c += [f"{p}_zero_crossings_14d", f"{p}_direction_streak", f"{p}_reversal_3d"]
    # N. Regime/Breakout (4)
    c += [f"{p}_band_position", f"{p}_band_width", f"{p}_drawdown_14d", f"{p}_breakout_strength"]
    # O. Daily Momentum (4)
    c += [f"{p}_daily_return_1d", f"{p}_daily_returns_3d_avg"]
    c += [f"{p}_momentum_divergence", f"{p}_daily_accel_1d"]
    return c  # 65개


FEAT_COLS = []
for _p in SIGNALS:
    FEAT_COLS += _sig_cols(_p)

# cross (old 8)
FEAT_COLS += [
    "click_search_ratio", "click_lead_7d", "click_lead_14d", "click_search_pos_gap",
    "blog_lead_7d", "blog_search_ratio",
    "instagram_lead_7d", "instagram_search_ratio",
]
# cross (new 12) — 교차상관 시차, 방향 동조, 기울기 괴리
FEAT_COLS += [
    "blog_search_best_lag", "blog_search_peak_xcorr",
    "insta_search_best_lag", "insta_search_peak_xcorr",
    "click_search_best_lag", "click_search_peak_xcorr",
    "blog_search_dir_agree", "insta_search_dir_agree", "click_search_dir_agree",
    "blog_search_slope_gap", "insta_search_slope_gap",
    "multi_slope_count",
]
# surge (10: intensity×2 + accel×2 + lead_days per platform)
FEAT_COLS += [
    "blog_surge_intensity_3d", "blog_surge_intensity_7d",
    "blog_surge_accel_3d", "blog_surge_accel_7d", "blog_surge_lead_days",
    "instagram_surge_intensity_3d", "instagram_surge_intensity_7d",
    "instagram_surge_accel_3d", "instagram_surge_accel_7d", "instagram_surge_lead_days",
]
# 멀티채널 종합 (19: conversion_rate×5 + channel_active×5 + accel×4 + buzz_zscore×5)
_W_MULTI = [1, 3, 7, 14, 30]
for _n in _W_MULTI:
    FEAT_COLS += [f"past_conversion_rate_{_n}d", f"past_channel_active_{_n}d"]
    if _n >= 3:
        FEAT_COLS.append(f"multi_accel_{_n}d")
    FEAT_COLS.append(f"past_buzz_zscore_{_n}d")
# calendar (day_of_week 제거)
FEAT_COLS += ["month"]

LABEL_COLS = [
    "future_intensity", "future_acceleration", "peak_timing",
    "signal_agreement", "search_click_convergence",
    "buzz_composite", "momentum_score",
]
assert len(FEAT_COLS) == 310, f"expected 310, got {len(FEAT_COLS)}"


# ── main ──────────────────────────────────────────────────────────────────────

def _process_keyword(args):
    kw, s_arr, c_arr, plats, pred_idx, dates, s_inv, stride, cached_sf = args
    rows, skipped = [], 0
    for ti in pred_idx[::stride]:
        lo, hi = ti - LB, ti + FW
        if lo < 0 or hi > len(s_arr):
            skipped += 1
            continue
        if s_inv[lo:lo + 30].all():
            skipped += 1
            continue
        s_lb = s_arr[lo:ti]
        if float(np.std(s_lb)) == 0:
            skipped += 1
            continue
        c_lb = c_arr[lo:ti]

        # sig_feat: 캐시에서 조회하거나 직접 계산
        if cached_sf is not None and ti in cached_sf:
            row = dict(cached_sf[ti])
        else:
            sf = sig_feat(s_lb, "search")
            cf = sig_feat(c_lb, "click")
            row = {}
            row.update(sf)
            row.update(cf)
            for p, arr in plats.items():
                row.update(sig_feat(arr[lo:ti], p))

        # cross_feat
        sf = {k: v for k, v in row.items() if k.startswith("search_") or k == "days_since_search_max_14d"}
        cf = {k: v for k, v in row.items() if k.startswith("click_") or k == "days_since_click_max_14d"}
        pfs = {}
        for p in ["blog", "instagram"]:
            pfs[p] = {k: v for k, v in row.items() if k.startswith(f"{p}_") or k == f"days_since_{p}_max_14d"}

        plat_lbs = {p: plats[p][lo:ti] for p in plats}
        all_lbs = {"search": s_lb, "click": c_lb}
        all_lbs.update(plat_lbs)
        row.update(cross_feat(sf, cf, pfs, all_lbs))

        # surge
        for pp in ["blog", "instagram"]:
            p_lb = plat_lbs.get(pp, np.zeros(LB, dtype=np.float64))
            p_base = float(np.mean(p_lb[:30]))
            for sn in [3, 7]:
                if p_base > EPS:
                    row[f"{pp}_surge_intensity_{sn}d"] = float(np.clip(np.max(p_lb[-sn:]) / p_base - 1, 0, 50))
                else:
                    row[f"{pp}_surge_intensity_{sn}d"] = 0.0
                if p_base > EPS:
                    row[f"{pp}_surge_accel_{sn}d"] = float(np.polyfit(np.arange(sn), p_lb[-sn:], 1)[0]) / p_base
                else:
                    row[f"{pp}_surge_accel_{sn}d"] = 0.0
            p_peak = int(np.argmax(p_lb[-14:]))
            s_peak = int(np.argmax(s_lb[-14:]))
            row[f"{pp}_surge_lead_days"] = s_peak - p_peak

        # calendar + labels
        row["keyword"] = kw
        dt = dates[ti]
        row["date"] = dt.date()
        row["month"] = dt.month

        s_fw, c_fw = s_arr[ti:hi], c_arr[ti:hi]
        b_lb = plats.get("blog", np.zeros(LB))[lo:ti]
        b_fw = plats.get("blog", np.zeros(len(s_arr)))[ti:hi]
        i_lb = plats.get("instagram", np.zeros(LB))[lo:ti]
        i_fw = plats.get("instagram", np.zeros(len(s_arr)))[ti:hi]
        row.update(compute_labels(s_fw, c_fw, b_fw, i_fw, s_lb, c_lb, b_lb, i_lb))
        rows.append(row)
    return rows, skipped


def _build_signals(search, click, mention, kws, dates):
    """L1: 키워드별 전처리된 시그널 배열 생성 또는 캐시 로드"""
    import h5py
    meta = _load_meta()
    h5_path = CACHE_DIR / "signals.h5"

    if _l1_valid(meta):
        print("  [L1 cache hit] signals.h5 로드")
        signals = {}
        with h5py.File(h5_path, "r") as f:
            for kw in kws:
                g = f[kw]
                signals[kw] = {
                    "s": g["search"][:], "c": g["click"][:],
                    "s_inv": g["s_inv"][:].astype(bool),
                    "plats": {p: g[p][:] for p in ["blog", "instagram"]},
                }
        return signals

    print("  [L1 cache miss] fill_sig 실행")
    signals = {}
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
        signals[kw] = {"s": s, "c": c, "s_inv": s_inv, "plats": plats}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "w") as f:
        for kw, sig in signals.items():
            g = f.create_group(kw)
            g.create_dataset("search", data=sig["s"])
            g.create_dataset("click", data=sig["c"])
            g.create_dataset("s_inv", data=sig["s_inv"].astype(np.uint8))
            for p in ["blog", "instagram"]:
                g.create_dataset(p, data=sig["plats"].get(p, np.zeros(len(dates))))

    meta["l1_raw_mtime"] = _raw_mtime()
    _save_meta(meta)
    print(f"  [L1 cache saved] {h5_path}")
    return signals


def _sig_feat_worker(args):
    """L2 워커: 키워드 하나의 sig_feat 계산"""
    kw, s, c, plats, s_inv, pred_idx, stride = args
    rows = []
    for ti in pred_idx[::stride]:
        lo = ti - LB
        if lo < 0 or ti + FW > len(s):
            continue
        if s_inv[lo:lo + 30].all():
            continue
        s_lb = s[lo:ti]
        if float(np.std(s_lb)) == 0:
            continue
        c_lb = c[lo:ti]
        sf = sig_feat(s_lb, "search")
        cf = sig_feat(c_lb, "click")
        row = {"keyword": kw, "date_idx": ti}
        row.update(sf)
        row.update(cf)
        for p, arr in plats.items():
            row.update(sig_feat(arr[lo:ti], p))
        rows.append(row)
    return rows


def _build_sig_feats(signals, kws, dates, pred_idx, stride):
    """L2: sig_feat 결과 캐시 로드 또는 재계산 (multiprocessing)"""
    from multiprocessing import Pool, cpu_count
    meta = _load_meta()
    pq_path = CACHE_DIR / "sig_feat.parquet"

    if _l2_valid(meta):
        print("  [L2 cache hit] sig_feat.parquet 로드")
        return pd.read_parquet(pq_path)

    n_workers = max(1, cpu_count() - 1)
    print(f"  [L2 cache miss] sig_feat 재계산 (workers={n_workers})")

    def gen_l2_tasks():
        for kw in kws:
            sig = signals[kw]
            yield (kw, sig["s"], sig["c"], sig["plats"], sig["s_inv"], pred_idx, stride)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    chunks, total = [], 0
    CHUNK_FLUSH = 50  # 키워드 50개마다 parquet flush
    part_files = []
    with Pool(n_workers, maxtasksperchild=50) as pool:
        for i, kw_rows in enumerate(pool.imap_unordered(_sig_feat_worker, gen_l2_tasks(), chunksize=10), 1):
            chunks.extend(kw_rows)
            total += len(kw_rows)
            print(f"    sig_feat {i}/{len(kws)} ({total:,})")
            if len(chunks) >= CHUNK_FLUSH * 1300:
                part = CACHE_DIR / f"sig_feat_part{len(part_files)}.parquet"
                pd.DataFrame(chunks).to_parquet(part, index=False)
                part_files.append(part)
                chunks.clear()
    if chunks:
        part = CACHE_DIR / f"sig_feat_part{len(part_files)}.parquet"
        pd.DataFrame(chunks).to_parquet(part, index=False)
        part_files.append(part)
        chunks.clear()

    df = pd.concat([pd.read_parquet(p) for p in part_files], ignore_index=True)
    df.to_parquet(pq_path, index=False)
    for p in part_files:
        p.unlink()
    meta["l2_sig_feat_hash"] = _sig_feat_hash()
    _save_meta(meta)
    print(f"  [L2 cache saved] {pq_path} ({len(df):,} rows)")
    return df


def main():
    import argparse
    from multiprocessing import Pool, cpu_count

    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=SAMPLE_STRIDE, help="샘플링 간격 (기본 1일)")
    parser.add_argument("--no-cache", action="store_true", help="캐시 무시하고 전체 재생성")
    parser.add_argument("--clear-cache", action="store_true", help="캐시 삭제 후 재생성")
    args = parser.parse_args()
    stride = args.stride

    if args.clear_cache:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print("  [cache cleared]")

    if args.no_cache or args.clear_cache:
        # 캐시 무효화
        if CACHE_META.exists():
            CACHE_META.unlink()

    print("=" * 50)
    print(f"building dataset ({len(FEAT_COLS)} features, stride={stride})")
    print("=" * 50)

    search, click, mention = load_all()

    kws = search.index.intersection(click.index)
    dates = search.columns.intersection(click.columns).sort_values()

    mask = (dates >= START) & (dates <= END)
    pred_idx = np.where(mask)[0]
    mention_kws = set(next(iter(mention.values())).index)

    est_samples = len(kws) * len(pred_idx[::stride])
    print(f"  {len(kws)} keywords, {len(dates)} days, {len(pred_idx)} pred dates")
    print(f"  stride={stride} → ~{est_samples:,} samples (before filter)")
    print(f"  mention: {len(kws.intersection(mention_kws))}/{len(kws)}")

    # L1: 전처리 시그널
    signals = _build_signals(search, click, mention, kws, dates)
    del search, click, mention  # raw DataFrame 해제

    # L2: sig_feat 캐시 (parquet 저장만, 메모리에 적재하지 않음)
    _build_sig_feats(signals, kws, dates, pred_idx, stride)

    # L3: cross_feat + surge + labels (multiprocessing)
    n_workers = max(1, cpu_count() - 1)
    print(f"  workers: {n_workers}")

    def gen_tasks():
        for kw in kws:
            sig = signals[kw]
            yield (kw, sig["s"], sig["c"], sig["plats"], pred_idx, dates, sig["s_inv"], stride, None)

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
    ALL_LABELS = LABEL_COLS
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
