# LSTM 데이터셋 빌드 — 60일 z-score 시퀀스 + 보조 피처 + 13 라벨 + 궤적 타겟
# seq_y를 같은 스크립트에서 생성해 순서 정합성 보장 (build_seq_y.py 대체)
# 출력: lstm_X.npy, lstm_feat.npy, lstm_labels.npy, lstm_seq_y.npy, lstm_meta.csv

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import (
    ROOT, START, END, LB, FW, EPS,
    LSTM_SAMPLE_STRIDE, LSTM_LABEL_COLS, LSTM_FEAT_COLS,
)
from pipeline.features.build_dataset import load_all, _build_signals

# 출력 경로
OUT_DIR    = ROOT / "data" / "processed"
OUT_X      = OUT_DIR / "lstm_X.npy"
OUT_FEAT   = OUT_DIR / "lstm_feat.npy"
OUT_LABELS = OUT_DIR / "lstm_labels.npy"
OUT_SEQ_Y  = OUT_DIR / "lstm_seq_y.npy"
OUT_META   = OUT_DIR / "lstm_meta.csv"


# 13개 라벨 생성 — forward window 정보 기반, lookback 참조 최소화
def compute_lstm_labels(combined_fw, combined_lb):
    mean_lb = float(np.mean(combined_lb)) + EPS
    labels = {}

    # 규모 — 미래 N일 4채널 합산 평균의 log 스케일
    labels["fw_magnitude_5d"]  = float(np.log1p(np.mean(combined_fw[:5])))
    labels["fw_magnitude_10d"] = float(np.log1p(np.mean(combined_fw[:10])))
    labels["fw_magnitude_15d"] = float(np.log1p(np.mean(combined_fw[:15])))

    # 성장 — 과거 평균 대비 미래 평균 비율
    labels["fw_growth_10d"] = float(np.clip(np.mean(combined_fw[:10]) / mean_lb, 0, 50))
    labels["fw_growth_15d"] = float(np.clip(np.mean(combined_fw[:15]) / mean_lb, 0, 50))

    # 피크 위치 — forward window 내 최고점 위치 (0=초반, 1=후반)
    labels["fw_peak_pos_10d"] = float(np.argmax(combined_fw[:10])) / 9.0
    labels["fw_peak_pos_15d"] = float(np.argmax(combined_fw[:15])) / 14.0

    # 스파이크 강도 — 최고점이 평균 대비 얼마나 뾰족한가
    mean_10 = float(np.mean(combined_fw[:10])) + EPS
    mean_15 = float(np.mean(combined_fw[:15])) + EPS
    labels["fw_spike_10d"] = float(np.clip(np.max(combined_fw[:10]) / mean_10, 1, 50))
    labels["fw_spike_15d"] = float(np.clip(np.max(combined_fw[:15]) / mean_15, 1, 50))

    # 변동성 — forward window 내 변동계수
    labels["fw_cv_10d"] = float(np.clip(np.std(combined_fw[:10]) / mean_10, 0, 10))
    labels["fw_cv_15d"] = float(np.clip(np.std(combined_fw[:15]) / mean_15, 0, 10))

    # 하락 — 피크 대비 마지막 날의 하락률
    max_10 = float(np.max(combined_fw[:10])) + EPS
    max_15 = float(np.max(combined_fw[:15])) + EPS
    labels["fw_decline_10d"] = float(np.clip((max_10 - float(combined_fw[9]))  / max_10, 0, 1))
    labels["fw_decline_15d"] = float(np.clip((max_15 - float(combined_fw[14])) / max_15, 0, 1))

    return labels


# 보조 피처 — 시퀀스 z-score 정규화로 사라진 정보 보충
def compute_lstm_features(s_lb, c_lb, b_lb, i_lb, month):
    f = {}

    # 스케일 — 절대 규모
    f["scale_search"] = float(np.mean(s_lb))
    f["scale_click"]  = float(np.mean(c_lb))
    f["scale_blog"]   = float(np.mean(b_lb))
    f["scale_insta"]  = float(np.mean(i_lb))
    f["scale_total"]  = f["scale_search"] + f["scale_click"] + f["scale_blog"] + f["scale_insta"]

    # 변동성 — 채널별 노이즈 수준
    f["vol_search"] = float(np.std(s_lb))
    f["vol_click"]  = float(np.std(c_lb))
    f["vol_blog"]   = float(np.std(b_lb))
    f["vol_insta"]  = float(np.std(i_lb))

    # 채널 관계 — 채널 간 비율과 활성 채널 수
    f["click_search_ratio"]  = f["scale_click"] / (f["scale_search"] + EPS)
    f["social_search_ratio"] = (f["scale_blog"] + f["scale_insta"]) / (f["scale_search"] + EPS)
    f["blog_insta_ratio"]    = f["scale_blog"] / (f["scale_insta"] + EPS)
    f["active_channels"]     = float(sum(
        1 for m in [f["scale_search"], f["scale_click"], f["scale_blog"], f["scale_insta"]]
        if m > 1.0
    ))

    # 캘린더 — 주기적 월 인코딩
    f["month_sin"] = float(np.sin(2 * np.pi * month / 12))
    f["month_cos"] = float(np.cos(2 * np.pi * month / 12))

    # 맥락 — 전체 트렌드와 현재 상태
    combined_lb = s_lb + c_lb + b_lb + i_lb
    f["total_trend"] = float(np.polyfit(np.arange(LB), combined_lb, 1)[0])
    slope_7  = float(np.polyfit(np.arange(7),  combined_lb[-7:],  1)[0])
    slope_30 = float(np.polyfit(np.arange(30), combined_lb[-30:], 1)[0])
    f["recent_accel"]   = slope_7 - slope_30
    f["peak_recency"]   = float(np.argmax(combined_lb)) / (LB - 1)
    f["activity_ratio"] = float(np.sum(s_lb > 0)) / LB

    return f


# 키워드 하나의 전체 예측 시점에서 시퀀스, 피처, 라벨, 궤적 타겟 생성
def _process_keyword(args):
    kw, s_arr, c_arr, plats, pred_idx, dates, s_inv, stride = args

    X_rows     = []
    feat_rows  = []
    label_rows = []
    seq_y_rows = []
    meta_rows  = []
    skipped    = 0

    blog_arr  = plats.get("blog",      np.zeros(len(s_arr), dtype=np.float64))
    insta_arr = plats.get("instagram", np.zeros(len(s_arr), dtype=np.float64))

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
        b_lb = blog_arr[lo:ti]
        i_lb = insta_arr[lo:ti]

        # 입력 시퀀스 — 채널별 z-score 정규화
        channels = [s_lb, c_lb, b_lb, i_lb]
        seq = np.zeros((LB, 4), dtype=np.float32)
        for ch_idx, ch in enumerate(channels):
            ch_mean = float(np.mean(ch))
            ch_std  = float(np.std(ch)) + EPS
            seq[:, ch_idx] = ((ch - ch_mean) / ch_std).astype(np.float32)

        # 보조 피처
        month = dates[ti].month
        feat = compute_lstm_features(s_lb, c_lb, b_lb, i_lb, month)
        feat_vec = np.array([feat[col] for col in LSTM_FEAT_COLS], dtype=np.float32)

        # forward window
        s_fw = s_arr[ti:hi]
        c_fw = c_arr[ti:hi]
        b_fw = blog_arr[ti:hi]
        i_fw = insta_arr[ti:hi]

        combined_fw = s_fw + c_fw + b_fw + i_fw
        combined_lb = s_lb + c_lb + b_lb + i_lb

        # 13개 라벨
        lbl = compute_lstm_labels(combined_fw, combined_lb)
        lbl_vec = np.array([lbl[col] for col in LSTM_LABEL_COLS], dtype=np.float32)

        # 궤적 타겟 — 합산 신호의 z-score 정규화 (역변환용 mean/std 메타에 저장)
        mean_c = float(np.mean(combined_lb))
        std_c  = float(np.std(combined_lb)) + EPS
        seq_y  = np.clip((combined_fw - mean_c) / std_c, -10, 10).astype(np.float32)

        X_rows.append(seq)
        feat_rows.append(feat_vec)
        label_rows.append(lbl_vec)
        seq_y_rows.append(seq_y)
        meta_rows.append({
            "keyword":    kw,
            "date":       dates[ti].date(),
            "seq_y_mean": round(mean_c, 6),
            "seq_y_std":  round(std_c, 6),
        })

    return X_rows, feat_rows, label_rows, seq_y_rows, meta_rows, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride",  type=int, default=LSTM_SAMPLE_STRIDE)
    parser.add_argument("--workers", type=int, default=cpu_count())
    parser.add_argument("--no-cache",    action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    if args.clear_cache:
        import shutil
        from pipeline.config import CACHE_DIR
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print("  [cache cleared]")

    if args.no_cache or args.clear_cache:
        from pipeline.config import CACHE_META
        if CACHE_META.exists():
            CACHE_META.unlink()

    n_feat  = len(LSTM_FEAT_COLS)
    n_label = len(LSTM_LABEL_COLS)

    print("=" * 55)
    print(f"LSTM dataset build (stride={args.stride}, workers={args.workers})")
    print(f"  X      : (N, {LB}, 4)  z-score per channel")
    print(f"  feat   : (N, {n_feat})")
    print(f"  labels : (N, {n_label})")
    print(f"  seq_y  : (N, {FW})  z-score trajectory")
    print("=" * 55)

    search, click, mention = load_all()

    kws   = search.index.intersection(click.index)
    dates = search.columns.intersection(click.columns).sort_values()

    mask     = (dates >= START) & (dates <= END)
    pred_idx = np.where(mask)[0]

    print(f"  {len(kws)} keywords  {len(dates)} days  {len(pred_idx)} pred dates")
    print(f"  stride={args.stride} -> ~{len(kws) * len(pred_idx[::args.stride]):,} (before filter)")

    signals = _build_signals(search, click, mention, kws, dates)
    del search, click, mention

    def gen_tasks():
        for kw in kws:
            sig = signals[kw]
            yield (kw, sig["s"], sig["c"], sig["plats"], pred_idx, dates, sig["s_inv"], args.stride)

    all_X, all_feat, all_labels, all_seq_y, all_meta = [], [], [], [], []
    total, skipped = 0, 0

    print(f"  workers: {args.workers}")
    with Pool(args.workers, maxtasksperchild=50) as pool:
        for i, result in enumerate(pool.imap(_process_keyword, gen_tasks(), chunksize=10), 1):
            X_rows, feat_rows, label_rows, seq_y_rows, meta_rows, kw_skipped = result
            if X_rows:
                all_X.extend(X_rows)
                all_feat.extend(feat_rows)
                all_labels.extend(label_rows)
                all_seq_y.extend(seq_y_rows)
                all_meta.extend(meta_rows)
                total += len(X_rows)
            skipped += kw_skipped
            if i % 50 == 0 or i == len(kws):
                print(f"  {i}/{len(kws)} ({total:,} samples)")

    print(f"\n  {total:,} samples ({skipped:,} skipped)")

    X      = np.stack(all_X,      axis=0)
    feat   = np.stack(all_feat,   axis=0)
    labels = np.stack(all_labels, axis=0)
    seq_y  = np.stack(all_seq_y,  axis=0)
    meta   = pd.DataFrame(all_meta)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_X,      X)
    np.save(OUT_FEAT,   feat)
    np.save(OUT_LABELS, labels)
    np.save(OUT_SEQ_Y,  seq_y)
    meta.to_csv(OUT_META, index=False, encoding="utf-8-sig")

    print(f"\n  X      : {X.shape}  {X.dtype}")
    print(f"  feat   : {feat.shape}  {feat.dtype}")
    print(f"  labels : {labels.shape}  {labels.dtype}")
    print(f"  seq_y  : {seq_y.shape}  {seq_y.dtype}")

    for i, col in enumerate(LSTM_LABEL_COLS):
        v = labels[:, i]
        print(f"  {col:25s}  mean={v.mean():.4f}  std={v.std():.4f}  min={v.min():.4f}  max={v.max():.4f}")

    print(f"\n  seq_y  mean={seq_y.mean():.4f}  std={seq_y.std():.4f}  min={seq_y.min():.4f}  max={seq_y.max():.4f}")

    for f_path in [OUT_X, OUT_FEAT, OUT_LABELS, OUT_SEQ_Y, OUT_META]:
        print(f"  saved -> {f_path}")


if __name__ == "__main__":
    main()
