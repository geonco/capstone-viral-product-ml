# seq2seq 타겟 생성 — 미래 15일 4채널 합산 시계열 (lookback 평균으로 정규화)
# 기존 lstm_X.npy와 순서 동일하게 생성
# 출력: lstm_seq_y.npy (N, 15)  값 1.0 = 과거 평균 수준, 2.0 = 2배 상승

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import (
    ROOT, START, END, LB, FW, SAMPLE_STRIDE, EPS,
)
from pipeline.features.build_dataset import load_all, _build_signals

DATA_DIR   = ROOT / "data" / "processed"
OUT_SEQ_Y  = DATA_DIR / "lstm_seq_y.npy"


def _process_keyword(args):
    # 키워드 하나의 전체 예측 시점에서 정규화된 forward 15일 시계열 생성
    kw, s_arr, c_arr, plats, pred_idx, dates, s_inv, stride = args

    seq_y_rows = []
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

        # lookback 4채널 합산 평균 — 정규화 기준
        combined_lb_mean = (
            float(np.mean(s_lb))
            + float(np.mean(c_arr[lo:ti]))
            + float(np.mean(blog_arr[lo:ti]))
            + float(np.mean(insta_arr[lo:ti]))
        ) + EPS

        # forward 15일 4채널 합산 시계열
        fw_combined = (
            s_arr[ti:hi]
            + c_arr[ti:hi]
            + blog_arr[ti:hi]
            + insta_arr[ti:hi]
        )

        # lookback 평균으로 정규화 — 1.0=과거 평균 수준, 2.0=2배 상승
        seq_y = (fw_combined / combined_lb_mean).astype(np.float32)
        seq_y_rows.append(seq_y)

    return seq_y_rows, skipped


def main():
    # meta 로드해서 기존 lstm_X.npy와 동일한 순서·필터 조건 보장
    meta = pd.read_csv(DATA_DIR / "lstm_meta.csv")
    print(f"기존 샘플 수: {len(meta):,}")

    print("=" * 50)
    print(f"building lstm_seq_y (N, {FW})  workers={cpu_count()}")
    print("=" * 50)

    search, click, mention = load_all()

    kws   = search.index.intersection(click.index)
    dates = search.columns.intersection(click.columns).sort_values()

    mask     = (dates >= START) & (dates <= END)
    pred_idx = np.where(mask)[0]

    signals = _build_signals(search, click, mention, kws, dates)
    del search, click, mention

    def gen_tasks():
        for kw in kws:
            sig = signals[kw]
            yield (kw, sig["s"], sig["c"], sig["plats"], pred_idx, dates, sig["s_inv"], SAMPLE_STRIDE)

    all_seq_y = []
    total, skipped = 0, 0

    with Pool(cpu_count(), maxtasksperchild=50) as pool:
        for i, (seq_y_rows, kw_skipped) in enumerate(
            pool.imap_unordered(_process_keyword, gen_tasks(), chunksize=10), 1
        ):
            if seq_y_rows:
                all_seq_y.extend(seq_y_rows)
                total += len(seq_y_rows)
            skipped += kw_skipped
            print(f"  {i}/{len(kws)} ({total:,} samples)")

    print(f"\n  {total:,} samples ({skipped:,} skipped)")

    seq_y = np.stack(all_seq_y, axis=0)  # (N, 15)  float32

    # 기존 lstm_X.npy와 샘플 수 일치 확인
    if len(seq_y) != len(meta):
        print(f"[WARN] 샘플 수 불일치: seq_y={len(seq_y)}, meta={len(meta)}")
    else:
        print(f"[OK] 샘플 수 일치: {len(seq_y):,}")

    np.save(OUT_SEQ_Y, seq_y)
    print(f"\n  seq_y shape : {seq_y.shape}  dtype={seq_y.dtype}")
    print(f"  mean={seq_y.mean():.4f}  std={seq_y.std():.4f}  min={seq_y.min():.4f}  max={seq_y.max():.2f}")
    print(f"  saved -> {OUT_SEQ_Y}")


if __name__ == "__main__":
    main()
