# LSTM 데이터셋 빌드 — 60일 lookback × 4채널 시계열 → peak_time 3-class + 15개 회귀 라벨
# 출력: lstm_X.npy (N, 60, 4), lstm_y.npy (N,), lstm_labels.npy (N, 15), lstm_meta.csv

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import (
    ROOT, START, END, LB, FW, SAMPLE_STRIDE, EPS,
    SIGNAL_WEIGHTS, LABEL_COLS,
)
from pipeline.features.build_dataset import load_all, _build_signals, compute_labels

# 출력 경로
OUT_X      = ROOT / "data" / "processed" / "lstm_X.npy"
OUT_Y      = ROOT / "data" / "processed" / "lstm_y.npy"
OUT_LABELS = ROOT / "data" / "processed" / "lstm_labels.npy"
OUT_META   = ROOT / "data" / "processed" / "lstm_meta.csv"

# peak_time 구간 경계 — 0-indexed (argmax 기준)
# class 0: peak_day 0~4   → 미래 1~5일 내 피크
# class 1: peak_day 5~9   → 미래 6~10일 내 피크
# class 2: peak_day 10~14 → 미래 11~15일 내 피크
PEAK_BIN_EDGES = [5, 10]


def _peak_class(fw_combined):
    # 4채널 합산 forward window에서 피크 위치를 3-class로 변환
    peak_day = int(np.argmax(fw_combined))
    if peak_day < PEAK_BIN_EDGES[0]:
        return 0
    elif peak_day < PEAK_BIN_EDGES[1]:
        return 1
    else:
        return 2


def _process_keyword(args):
    # 키워드 하나의 전체 예측 시점을 순회하며 (시퀀스, peak_class, 15라벨, 메타) 생성
    kw, s_arr, c_arr, plats, pred_idx, dates, s_inv, stride = args

    X_rows      = []
    y_rows      = []
    label_rows  = []
    meta_rows   = []
    skipped     = 0

    blog_arr  = plats.get("blog",      np.zeros(len(s_arr), dtype=np.float64))
    insta_arr = plats.get("instagram", np.zeros(len(s_arr), dtype=np.float64))

    for ti in pred_idx[::stride]:
        lo, hi = ti - LB, ti + FW
        if lo < 0 or hi > len(s_arr):
            skipped += 1
            continue

        # lookback 앞 30일 전부 비활성이면 제외
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

        # 입력 시퀀스 — 채널별 lookback 평균으로 정규화해 상대 패턴만 남김
        channels = [s_lb, c_lb, b_lb, i_lb]
        seq = np.zeros((LB, 4), dtype=np.float32)
        for ch_idx, ch in enumerate(channels):
            baseline = float(np.mean(ch)) + EPS
            seq[:, ch_idx] = (ch / baseline).astype(np.float32)

        # forward window
        s_fw = s_arr[ti:hi]
        c_fw = c_arr[ti:hi]
        b_fw = blog_arr[ti:hi]
        i_fw = insta_arr[ti:hi]

        # peak_class — 4채널 합산 argmax 기반 3-class
        fw_combined = s_fw + c_fw + b_fw + i_fw
        peak_day    = int(np.argmax(fw_combined))
        label       = _peak_class(fw_combined)

        # 15개 회귀 라벨 — LightGBM과 동일한 compute_labels 재사용
        lbl_dict = compute_labels(s_fw, c_fw, b_fw, i_fw, s_lb, c_lb, b_lb, i_lb)
        lbl_vec  = np.array([lbl_dict[col] for col in LABEL_COLS], dtype=np.float32)

        X_rows.append(seq)
        y_rows.append(label)
        label_rows.append(lbl_vec)
        meta_rows.append({
            "keyword":    kw,
            "date":       dates[ti].date(),
            "peak_day":   peak_day,
            "peak_class": label,
        })

    return X_rows, y_rows, label_rows, meta_rows, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride",  type=int, default=SAMPLE_STRIDE, help="샘플링 간격 (기본 1일)")
    parser.add_argument("--workers", type=int, default=cpu_count(),   help="멀티프로세싱 코어 수 (기본 전체)")
    parser.add_argument("--no-cache",    action="store_true", help="L1 신호 캐시 무시")
    parser.add_argument("--clear-cache", action="store_true", help="L1 신호 캐시 삭제 후 재생성")
    args = parser.parse_args()

    if args.clear_cache:
        import shutil
        from pipeline.config import CACHE_DIR, CACHE_META
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            print("  [cache cleared]")

    if args.no_cache or args.clear_cache:
        from pipeline.config import CACHE_META
        if CACHE_META.exists():
            CACHE_META.unlink()

    print("=" * 50)
    print(f"building LSTM dataset (stride={args.stride}, workers={args.workers})")
    print(f"  input   : (N, {LB}, 4)  lookback {LB}d x 4ch")
    print(f"  y       : 3-class  0=<=5d  1=6~10d  2=11~15d")
    print(f"  labels  : {len(LABEL_COLS)} regression labels (same as LightGBM)")
    print("=" * 50)

    search, click, mention = load_all()

    kws   = search.index.intersection(click.index)
    dates = search.columns.intersection(click.columns).sort_values()

    mask     = (dates >= START) & (dates <= END)
    pred_idx = np.where(mask)[0]

    print(f"  {len(kws)} keywords  {len(dates)} days  {len(pred_idx)} pred dates")
    print(f"  stride={args.stride} → ~{len(kws) * len(pred_idx[::args.stride]):,} samples (before filter)")

    # L1 캐시에서 전처리 신호 로드 또는 생성
    signals = _build_signals(search, click, mention, kws, dates)
    del search, click, mention

    def gen_tasks():
        for kw in kws:
            sig = signals[kw]
            yield (kw, sig["s"], sig["c"], sig["plats"], pred_idx, dates, sig["s_inv"], args.stride)

    all_X, all_y, all_labels, all_meta = [], [], [], []
    total, skipped = 0, 0

    print(f"  workers: {args.workers}")
    with Pool(args.workers, maxtasksperchild=50) as pool:
        for i, (X_rows, y_rows, label_rows, meta_rows, kw_skipped) in enumerate(
            pool.imap_unordered(_process_keyword, gen_tasks(), chunksize=10), 1
        ):
            if X_rows:
                all_X.extend(X_rows)
                all_y.extend(y_rows)
                all_labels.extend(label_rows)
                all_meta.extend(meta_rows)
                total += len(X_rows)
            skipped += kw_skipped
            print(f"  {i}/{len(kws)} ({total:,} samples)")

    print(f"\n  {total:,} samples ({skipped:,} skipped)")

    X      = np.stack(all_X,      axis=0)         # (N, 60, 4)  float32
    y      = np.array(all_y,      dtype=np.int8)  # (N,)        int8
    labels = np.stack(all_labels, axis=0)         # (N, 15)     float32
    meta   = pd.DataFrame(all_meta)

    # peak_class 분포 확인
    label_names = {0: "≤5d", 1: "6~10d", 2: "11~15d"}
    for cls in range(3):
        cnt = int((y == cls).sum())
        print(f"  class {cls} ({label_names[cls]}): {cnt:,}  ({cnt / len(y) * 100:.1f}%)")

    OUT_X.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_X,      X)
    np.save(OUT_Y,      y)
    np.save(OUT_LABELS, labels)
    meta.to_csv(OUT_META, index=False, encoding="utf-8-sig")

    print(f"\n  X shape      : {X.shape}  dtype={X.dtype}")
    print(f"  y shape      : {y.shape}  dtype={y.dtype}")
    print(f"  labels shape : {labels.shape}  dtype={labels.dtype}")
    print(f"  label cols   : {LABEL_COLS}")
    print(f"  saved → {OUT_X}")
    print(f"  saved → {OUT_Y}")
    print(f"  saved → {OUT_LABELS}")
    print(f"  saved → {OUT_META}")


if __name__ == "__main__":
    main()
