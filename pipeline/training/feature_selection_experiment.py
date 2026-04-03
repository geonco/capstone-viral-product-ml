"""
SHAP 기반 Feature 제거 실험 스크립트

training2의 SHAP 중요도를 기준으로 하위 N% feature를 제거했을 때
성능 변화를 자동으로 실험한다.

- log_virality_score, peak_time 각각 자체 SHAP 순위로 독립 실험
- training2 best_params 재사용 (재튜닝 없음)
- 제거 비율: 0%, 10%, 20%, 30%, 40%, 50%

사용법:
    # 최초 실행 (사용자명 등록)
    python modeling/feature_selection_experiment.py --user leehyn
    python modeling/feature_selection_experiment.py --data modeling/data/dataset_v2.csv --user leehyn
    python modeling/feature_selection_experiment.py --use_splits --user leehyn

    # 이후 생략 가능
    python modeling/feature_selection_experiment.py
    python modeling/feature_selection_experiment.py --use_splits

출력:
    outputs/0403_leehyn_N/results.csv     # 전체 실험 결과
    outputs/0403_leehyn_N/rmse_curve.png  # RMSE vs 제거비율 그래프
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(os.path.dirname(__file__))
from data_loader import load_csv
from feature_engineering.feature_config import FEATURE_COLS
from run_id import make_run_id, DIR_OUTPUTS, write_summary
from train import (
    DEFAULT_PARAMS,
    TARGET_TIMING,
    TARGET_VIRALITY,
    split_data,
)

warnings.filterwarnings("ignore")

# ── 상수 ──────────────────────────────────────────────────────────────────────
_ROOT         = os.path.join(os.path.dirname(__file__), "..")
DIR_TRAINING2 = os.path.join(_ROOT, "outputs", "0402_leehyn_2", "metrics")


def _get_experiment_dir(user_arg: str | None = None) -> str:
    """날짜_사용자명_번호 형식의 다음 실험 폴더 경로 반환"""
    return os.path.join(DIR_OUTPUTS, make_run_id(DIR_OUTPUTS, user_arg))
REMOVAL_RATIOS    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_ESTIMATORS_PEAK = 5000   # lr=0.0014 수렴을 위해 기본값(1000)보다 크게 설정


# ── 유틸리티 ──────────────────────────────────────────────────────────────────
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_shap(target: str) -> pd.DataFrame:
    path = os.path.join(DIR_TRAINING2, f"shap_importance_{target}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SHAP 파일 없음: {path}")
    return pd.read_csv(path)


def load_best_params(target: str) -> dict:
    path = os.path.join(DIR_TRAINING2, f"best_params_{target}.json")
    if not os.path.isfile(path):
        print(f"  [경고] best_params 없음, DEFAULT_PARAMS 사용: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    return saved.get("best_params", {})


def select_features(shap_df: pd.DataFrame, removal_ratio: float) -> list[str]:
    """SHAP 하위 removal_ratio 비율 feature를 제거한 feature 목록 반환"""
    n_total  = len(shap_df)
    n_remove = int(n_total * removal_ratio)

    if n_remove == 0:
        removed = []
    else:
        # shap_df는 mean_abs_shap 내림차순 정렬 가정
        removed = shap_df.tail(n_remove)["feature"].tolist()

    features = [f for f in FEATURE_COLS if f not in set(removed)]
    return features, removed


# ── 단일 실험 ─────────────────────────────────────────────────────────────────
def run_single(
    X_train, y_train,
    X_valid, y_valid,
    X_test,  y_test,
    features: list[str],
    target: str,
    params: dict,
) -> dict:
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train[features], y_train,
        eval_set=[(X_valid[features], y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    result = {"best_iteration": model.best_iteration_}
    for split, X, y in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test",  X_test,  y_test),
    ]:
        pred = model.predict(X[features])
        result[f"{split}_rmse"] = round(rmse(y, pred), 4)
        result[f"{split}_mae"]  = round(float(mean_absolute_error(y, pred)), 4)

    return result


# ── 타겟 루프 ─────────────────────────────────────────────────────────────────
def run_target_experiments(
    target: str,
    objective: str,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df:  pd.DataFrame,
) -> list[dict]:

    print(f"\n{'='*55}")
    print(f"  {target} — Feature 제거 실험")
    print(f"{'='*55}")

    shap_df     = load_shap(target)
    best_params = load_best_params(target)
    params      = {**DEFAULT_PARAMS, "objective": objective, **best_params}
    if target == TARGET_TIMING:
        params["n_estimators"] = N_ESTIMATORS_PEAK

    X_train = train_df[FEATURE_COLS]
    X_valid = valid_df[FEATURE_COLS]
    X_test  = test_df[FEATURE_COLS]
    y_train = train_df[target]
    y_valid = valid_df[target]
    y_test  = test_df[target]

    rows = []
    for ratio in REMOVAL_RATIOS:
        features, removed = select_features(shap_df, ratio)
        pct_label = f"{int(ratio * 100)}%"
        print(f"\n  제거 비율 {pct_label}  →  잔여 feature {len(features)}개 (제거 {len(removed)}개)")
        if removed:
            print(f"    제거된 feature: {removed[:5]}{'...' if len(removed) > 5 else ''}")

        result = run_single(
            X_train, y_train,
            X_valid, y_valid,
            X_test,  y_test,
            features, target, params,
        )
        print(
            f"    [test]  RMSE={result['test_rmse']:.4f}  MAE={result['test_mae']:.4f}"
            f"  (best_iter={result['best_iteration']})"
        )

        rows.append({
            "target"       : target,
            "removal_pct"  : int(ratio * 100),
            "n_features"   : len(features),
            "n_removed"    : len(removed),
            "removed_list" : "|".join(removed),
            "train_rmse"   : result["train_rmse"],
            "valid_rmse"   : result["valid_rmse"],
            "test_rmse"    : result["test_rmse"],
            "train_mae"    : result["train_mae"],
            "valid_mae"    : result["valid_mae"],
            "test_mae"     : result["test_mae"],
            "best_iter"    : result["best_iteration"],
        })

    return rows


# ── 시각화 ────────────────────────────────────────────────────────────────────
def plot_results(df: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, target in zip(axes, [TARGET_VIRALITY, TARGET_TIMING]):
        sub = df[df["target"] == target].sort_values("removal_pct")
        xs  = sub["removal_pct"].tolist()

        ax.plot(xs, sub["test_rmse"],  marker="o", label="Test RMSE")
        ax.plot(xs, sub["valid_rmse"], marker="s", linestyle="--", label="Valid RMSE")

        # 최적 지점 표시
        best_idx = sub["test_rmse"].idxmin()
        best_x   = sub.loc[best_idx, "removal_pct"]
        best_y   = sub.loc[best_idx, "test_rmse"]
        ax.axvline(best_x, color="red", linestyle=":", alpha=0.6)
        ax.annotate(
            f"best={best_x}%\nRMSE={best_y:.4f}",
            xy=(best_x, best_y),
            xytext=(best_x + 3, best_y + (best_y * 0.02)),
            fontsize=8,
            color="red",
        )

        ax.set_title(f"{target}")
        ax.set_xlabel("Feature 제거 비율 (%)")
        ax.set_ylabel("RMSE")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("SHAP 기반 Feature 제거 실험 — RMSE 변화", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n그래프 저장: {save_path}")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main(csv_path: str, use_splits: bool, user: str | None = None):
    DIR_OUT = _get_experiment_dir(user)
    os.makedirs(DIR_OUT, exist_ok=True)
    print(f"실험 폴더: {DIR_OUT}")

    # 데이터 로드
    if use_splits:
        print("분할된 데이터 사용: modeling/data/splits/")
        train_df = load_csv("modeling/data/splits/train.csv")
        valid_df = load_csv("modeling/data/splits/valid.csv")
        test_df  = load_csv("modeling/data/splits/test.csv")
    else:
        df = load_csv(csv_path)
        train_df, valid_df, test_df = split_data(df)

    print(f"train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")

    all_rows = []

    # log_virality_score 실험
    all_rows += run_target_experiments(
        target    = TARGET_VIRALITY,
        objective = "regression",
        train_df  = train_df,
        valid_df  = valid_df,
        test_df   = test_df,
    )

    # peak_time 실험
    all_rows += run_target_experiments(
        target    = TARGET_TIMING,
        objective = "regression_l1",
        train_df  = train_df,
        valid_df  = valid_df,
        test_df   = test_df,
    )

    # 결과 저장
    result_df = pd.DataFrame(all_rows)
    csv_path_out = os.path.join(DIR_OUT, "results.csv")
    result_df.to_csv(csv_path_out, index=False)
    print(f"\n결과 저장: {csv_path_out}")

    # 요약 출력
    print(f"\n{'='*55}")
    print("최종 결과 요약 (test RMSE 기준)")
    print("="*55)
    summary = result_df[["target", "removal_pct", "n_features", "test_rmse", "test_mae"]]
    print(summary.to_string(index=False))

    # 최적 비율 출력
    print(f"\n{'='*55}")
    print("최적 제거 비율")
    print("="*55)
    for target in [TARGET_VIRALITY, TARGET_TIMING]:
        sub  = result_df[result_df["target"] == target]
        best = sub.loc[sub["test_rmse"].idxmin()]
        print(
            f"  {target}: {int(best['removal_pct'])}% 제거  "
            f"(잔여 {int(best['n_features'])}개)  "
            f"test RMSE={best['test_rmse']:.4f}"
        )

    # 시각화
    plot_path = os.path.join(DIR_OUT, "rmse_curve.png")
    plot_results(result_df, plot_path)

    # 실험 요약 문서 저장
    now_str   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name  = os.path.basename(DIR_OUT)
    data_desc = "modeling/data/splits/ (train/valid/test.csv)" if use_splits else csv_path
    ref_dir   = os.path.relpath(DIR_TRAINING2, os.path.join(DIR_OUTPUTS, ".."))

    best_rows = []
    for target in [TARGET_VIRALITY, TARGET_TIMING]:
        sub  = result_df[result_df["target"] == target]
        best = sub.loc[sub["test_rmse"].idxmin()]
        best_rows.append(
            f"| {target} | {int(best['removal_pct'])}% | {int(best['n_features'])}개 | {best['test_rmse']:.4f} | {best['test_mae']:.4f} |"
        )
    best_table = "\n".join(best_rows)

    ratio_str = ", ".join(f"{int(r*100)}%" for r in REMOVAL_RATIOS)

    markdown = f"""# 실험 요약 — {run_name}

## 기본 정보
- 유형: Feature Selection 실험 (SHAP 기반 하위 feature 제거)
- 일시: {now_str}
- 데이터: {data_desc}
- 참조 실험 (SHAP / best_params): {ref_dir}
- 제거 비율: {ratio_str}

## 최적 제거 비율 (test RMSE 최소 기준)

| target | 제거 비율 | 잔여 feature | test RMSE | test MAE |
|--------|-----------|-------------|-----------|----------|
{best_table}

## 생성 파일
- results.csv
- rmse_curve.png
"""
    write_summary(DIR_OUT, markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="modeling/data/dataset_v2.csv",
        help="CSV 경로 (기본값: modeling/data/dataset_v2.csv)",
    )
    parser.add_argument(
        "--use_splits", action="store_true",
        help="modeling/data/splits/ 폴더의 분할 데이터 사용",
    )
    parser.add_argument(
        "--user", type=str, default=None,
        help="실험자 이름 (최초 1회 지정하면 outputs/.username에 저장됨)",
    )
    args = parser.parse_args()
    main(args.data, args.use_splits, args.user)
