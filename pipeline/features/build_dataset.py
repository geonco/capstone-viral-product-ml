import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent
RAW = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "processed" / "dataset.csv"

PATHS = {
    "search": RAW / "search" / "absolute.csv",
    "click": RAW / "shopping_click" / "absolute.csv",
    "mention": RAW / "sometrends" / "sometrend_mention_long.csv",
}

START = pd.Timestamp("2022-05-01")
END = pd.Timestamp("2026-02-14")
LB, FW = 60, 15
SAMPLE_STRIDE = 1  # 빌드 시 stride=1 (전체), train/valid 서브샘플링은 train.py에서
EPS = 1e-6
CLIP = {"level": (0, 30), "growth": (-2, 20), "cv": (0, 10), "ratio": (0, 50)}

W_FULL = [3, 5, 7, 14, 30]          # 내부 계산용 (accel에 growth_3d/5d 필요)
W_STAT = [7, 14, 30]                 # skew, kurt, cv, rsi (최소 7개 데이터포인트 필요)
MENTION_COLS = {"블로그": "blog", "인스타그램": "instagram"}
SIGNALS = ["search", "click", "blog", "instagram"]


# ── cache ─────────────────────────────────────────────────────────────────────

CACHE_DIR = ROOT / "data" / "cache"
CACHE_META = CACHE_DIR / "meta.json"



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


# ── signal features (73 per signal) ──────────────────────────────────────────

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

    # N. Regime/Breakout — 볼린저 밴드, 돌파 강도 (5)
    mu_30 = float(np.mean(lb[-30:]))
    upper = mu_30 + 2 * std_30
    lower = mu_30 - 2 * std_30
    band_range = upper - lower
    f[f"{p}_band_position"] = float(lb[-1] - lower) / (band_range + EPS)
    f[f"{p}_band_width"] = band_range / (mu_30 + EPS)
    f[f"{p}_drawdown_14d"] = (float(lb[-1]) - float(np.max(lb[-14:]))) / (float(np.max(lb[-14:])) + EPS)
    f[f"{p}_breakout_strength"] = (float(lb[-1]) - upper) / (std_30 + EPS)
    bp_3d_ago = (float(lb[-4]) - lower) / (band_range + EPS)
    f[f"{p}_band_approach"] = f[f"{p}_band_position"] - bp_3d_ago

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

    # P. Daily Change — 최근 7일 개별 변화율 (7)
    for d in range(1, 8):
        change = (float(lb[-d]) - float(lb[-(d + 1)])) / (abs(float(lb[-(d + 1)])) + EPS)
        f[f"{p}_change_{d}d_ago"] = float(np.clip(change, -10, 100))

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


# ── cross features (39 = 8 ratio/lead + 12 xcorr/slope + 19 multi-channel) ──

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

    # 신규 — 교차상관 시차, 방향 동조, 기울기 괴리, 시차 크기 괴리
    s_lb = all_lbs["search"]
    for other in ["blog", "instagram", "click"]:
        prefix = "insta" if other == "instagram" else other
        o_lb = all_lbs.get(other, np.zeros_like(s_lb))
        lag, xcorr = _xcorr_lag(o_lb, s_lb)
        f[f"{prefix}_search_best_lag"] = lag
        f[f"{prefix}_search_peak_xcorr"] = xcorr
        f[f"{prefix}_search_dir_agree"] = _dir_agree(o_lb, s_lb)

        # lead_magnitude — best_lag 시점에서 other가 search 대비 얼마나 앞서 있었나
        if lag > 0 and len(o_lb) > lag + 3:
            o_level = float(np.mean(o_lb[-(lag + 3):-lag])) / (float(np.mean(o_lb)) + EPS)
            s_level = float(np.mean(s_lb[-(lag + 3):-lag])) / (float(np.mean(s_lb)) + EPS)
            f[f"{prefix}_search_lead_magnitude"] = o_level / (s_level + EPS)
        else:
            f[f"{prefix}_search_lead_magnitude"] = 1.0

    # 기울기 괴리 (blog/insta → search 선행 여부)
    f["blog_search_slope_gap"] = pfs.get("blog", {}).get("blog_norm_slope_7d", 0.0) - sf["search_norm_slope_7d"]
    f["insta_search_slope_gap"] = pfs.get("instagram", {}).get("instagram_norm_slope_7d", 0.0) - sf["search_norm_slope_7d"]

    # 다채널 모멘텀 (양의 기울기 신호 수)
    f["multi_slope_count"] = sum(
        1 for sig in ["search", "click", "blog", "instagram"]
        if all_lbs.get(sig, np.zeros(1)).shape[0] > 7
        and float(np.polyfit(np.arange(7), all_lbs[sig][-7:], 1)[0]) > 0
    )

    # weighted_slope_strength — 기울기 크기까지 반영한 가중 모멘텀 강도
    weights = {"search": 0.4, "click": 0.3, "blog": 0.2, "instagram": 0.1}
    wss = 0.0
    for sig, w in weights.items():
        lb = all_lbs.get(sig, np.zeros(1))
        if len(lb) >= 7:
            ns = float(np.polyfit(np.arange(7), lb[-7:], 1)[0]) / (float(np.mean(lb)) + EPS)
            wss += w * ns
    f["weighted_slope_strength"] = wss

    # activation_spread — 채널 활성화 시차 (동시=0, 순차적=큰 값)
    first_days = []
    for sig in ["search", "click", "blog", "instagram"]:
        lb = all_lbs.get(sig, np.zeros(1))
        if len(lb) >= 14:
            med = float(np.median(lb))
            if med > EPS:
                above = np.where(lb[-14:] > med)[0]
                first_days.append(int(above[0]) if len(above) > 0 else 14)
            else:
                first_days.append(14)
        else:
            first_days.append(14)
    f["activation_spread"] = max(first_days) - min(first_days)

    # ── 멀티채널 종합 피처 (3종×5윈도우 + accel×4윈도우(n≥3) + conv_trend = 20개) ──
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

    # conv_trend — 전환율 추세 (최근 3일 vs 14일)
    f["conv_trend"] = f["past_conversion_rate_3d"] - f["past_conversion_rate_14d"]

    return f


# ── labels ────────────────────────────────────────────────────────────────────

def compute_labels(s_fw, c_fw, b_fw, i_fw, s_lb, c_lb, b_lb, i_lb, kw_stats=None):
    # 5종 × 3윈도우 = 15 라벨
    sigs_fw = {"search": s_fw, "click": c_fw, "blog": b_fw, "instagram": i_fw}
    sigs_lb = {"search": s_lb, "click": c_lb, "blog": b_lb, "instagram": i_lb}
    weights = {"search": 0.4, "click": 0.3, "blog": 0.2, "instagram": 0.1}
    combined = s_fw + c_fw + b_fw + i_fw

    labels = {}
    for w in [5, 10, 15]:
        fw_w = combined[:w]

        # intensity — 4채널 합산 평균
        labels[f"intensity_{w}d"] = round(float(np.mean(fw_w)), 6)

        # buzz_composite — 4채널 z-score 가중합 (lookback 기준)
        comp = 0.0
        for sig, wt in weights.items():
            mu = float(np.mean(sigs_lb[sig]))
            sigma = float(np.std(sigs_lb[sig]))
            fw_mean = float(np.mean(sigs_fw[sig][:w]))
            z = (fw_mean - mu) / (sigma + EPS) if sigma > EPS else 0.0
            comp += wt * z
        labels[f"buzz_composite_{w}d"] = round(comp, 6)

        # growth — 과거 대비 4채널 가중 성장률
        gr = 0.0
        for sig, wt in weights.items():
            past = float(np.mean(sigs_lb[sig][-w:]))
            future = float(np.mean(sigs_fw[sig][:w]))
            gr += wt * ((future - past) / (past + EPS))
        labels[f"growth_{w}d"] = round(float(np.clip(gr, -5, 50)), 6)

        # sustainability — 현재 대비 미래 평균 유지율 (>1 성장, <1 하락)
        current = float(fw_w[0])
        labels[f"sustainability_{w}d"] = round(float(np.mean(fw_w)) / (current + EPS), 6)

        # crash — 현재에서 최저점까지 하락률 (0=안빠짐, 양수=빠짐)
        cr = (current - float(np.min(fw_w))) / (current + EPS) if current > EPS else 0.0
        labels[f"crash_{w}d"] = round(float(np.clip(cr, 0, 10)), 6)

    return labels


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
    # N. Regime/Breakout (5)
    c += [f"{p}_band_position", f"{p}_band_width", f"{p}_drawdown_14d", f"{p}_breakout_strength"]
    c.append(f"{p}_band_approach")
    # O. Daily Momentum (4)
    c += [f"{p}_daily_return_1d", f"{p}_daily_returns_3d_avg"]
    c += [f"{p}_momentum_divergence", f"{p}_daily_accel_1d"]
    # P. Daily Change (7)
    for d in range(1, 8):
        c.append(f"{p}_change_{d}d_ago")
    return c  # 73개


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
    "blog_search_lead_magnitude", "insta_search_lead_magnitude", "click_search_lead_magnitude",
    "weighted_slope_strength", "activation_spread",
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
FEAT_COLS += ["conv_trend"]
# calendar
FEAT_COLS += ["month"]

LABEL_COLS = [
    "intensity_5d", "intensity_10d", "intensity_15d",
    "buzz_composite_5d", "buzz_composite_10d", "buzz_composite_15d",
    "growth_5d", "growth_10d", "growth_15d",
    "sustainability_5d", "sustainability_10d", "sustainability_15d",
    "crash_5d", "crash_10d", "crash_15d",
]
assert len(FEAT_COLS) == 348, f"expected 348, got {len(FEAT_COLS)}"


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

    out = OUT_PATH
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
