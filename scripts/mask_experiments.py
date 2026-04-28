#!/usr/bin/env python3
# 실험 7 — MASK_COMBOS 일괄 실행, 결과 summary.csv로 통합

import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.config import MASK_COMBOS, ROOT


OUT_DIR = Path(ROOT) / "outputs" / "mask_experiment_7"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT_DIR / "summary.csv"
COMBOS_JSON = OUT_DIR / "combos.json"
FAIL_LOG = OUT_DIR / "failures.log"


def _hms(seconds):
    # 초 → 시:분:초 — 진행 출력 포맷
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _collect_run(combo_name, t_start):
    # 해당 combo의 모든 타겟 run 디렉토리에서 evaluation_results.csv 수집
    pattern = f"{ROOT}/outputs/run_*_{combo_name}"
    rows = []
    for run_dir in sorted(glob.glob(pattern)):
        # 시작 이후 생성된 디렉토리만
        if Path(run_dir).stat().st_mtime < t_start:
            continue
        eval_csv = Path(run_dir) / "metrics" / "evaluation_results.csv"
        if eval_csv.exists():
            df = pd.read_csv(eval_csv)
            df["combo"] = combo_name
            df["run"] = Path(run_dir).name
            rows.append(df)
    return rows


def main():
    # combos.json — 조합 이름과 마스크 구성 기록 (분석 시 참조)
    with open(COMBOS_JSON, "w", encoding="utf-8") as f:
        json.dump(MASK_COMBOS, f, indent=2, ensure_ascii=False)

    combos = list(MASK_COMBOS.keys())
    n = len(combos)
    print(f"총 {n} 조합. 시작 시각 {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"출력 → {OUT_DIR}")

    all_rows = []
    failed = []
    start = time.time()

    for i, combo in enumerate(combos, 1):
        t0 = time.time()
        print(f"\n[{i:3d}/{n}] {combo}")
        try:
            subprocess.run(
                [".venv/bin/python3", "-m", "pipeline.training.train",
                 "--target", "all", "--train-stride", "3", "--combo", combo],
                check=True,
                cwd=str(ROOT),
                timeout=1800,
            )
            rows = _collect_run(combo, t0)
            all_rows.extend(rows)
            print(f"  수집된 타겟 수: {len(rows)}")
        except subprocess.TimeoutExpired:
            failed.append((combo, "timeout"))
            print(f"  타임아웃")
        except subprocess.CalledProcessError as e:
            failed.append((combo, f"returncode={e.returncode}"))
            print(f"  실패: returncode={e.returncode}")
        except Exception as e:
            failed.append((combo, str(e)))
            print(f"  예외: {e}")

        # incremental save — 중간 결과 실시간 확인용
        if all_rows:
            pd.concat(all_rows, ignore_index=True).to_csv(SUMMARY, index=False)

        dt = time.time() - t0
        elapsed = time.time() - start
        avg = elapsed / i
        remaining_est = avg * (n - i)
        print(f"  소요 {dt:.0f}s, 누적 {_hms(elapsed)}, 잔여 추정 {_hms(remaining_est)}")

    # 최종 저장
    if all_rows:
        df = pd.concat(all_rows, ignore_index=True)
        df.to_csv(SUMMARY, index=False)
        print(f"\n저장 완료: {SUMMARY}  rows={len(df)}")

    if failed:
        with open(FAIL_LOG, "w", encoding="utf-8") as f:
            for combo, err in failed:
                f.write(f"{combo}\t{err}\n")
        print(f"실패 {len(failed)}건 → {FAIL_LOG}")

    print(f"\n총 소요 시간: {_hms(time.time() - start)}")


if __name__ == "__main__":
    main()
