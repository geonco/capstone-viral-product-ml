# LightGBM 바이럴 예측 모델 (Optuna TPE 튜닝 포함)

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import (
    ROOT, FEAT_COLS,
    DATE_COL, TRAIN_RATIO, VALID_RATIO, GAP_DAYS,
    TARGET_CONFIG, DEFAULT_PARAMS, SEARCH_SPACE,
)
from pipeline.training.data_loader import load_csv

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def _make_run_dir(target: str = ""):
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = os.path.join(str(ROOT), "outputs", f"run_{stamp}_{target}" if target else f"run_{stamp}")
    dirs = {
        "figures": os.path.join(run, "figures"),
        "metrics": os.path.join(run, "metrics"),
        "models":  os.path.join(run, "models"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# 평가 지표 — RMSE, MAE, naive baseline 대비 gain
def rmse_score(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred, split: str, target: str) -> dict:
    rmse = rmse_score(y_true, y_pred)
    mae  = float(mean_absolute_error(y_true, y_pred))
    print(f"  [{split:5s}] {target}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    # test set에서 naive baseline 비교
    if split == "test":
        y_arr = np.array(y_true)
        bl_mean = float(mean_absolute_error(y_arr, np.full_like(y_arr, y_arr.mean())))
        bl_zero = float(mean_absolute_error(y_arr, np.zeros_like(y_arr)))
        best_bl = min(bl_mean, bl_zero)
        gain = (1 - mae / best_bl) * 100 if best_bl > 0 else 0
        print(f"         baseline MAE: mean={bl_mean:.4f} zero={bl_zero:.4f} → gain: {gain:+.1f}%")

    return {"split": split, "target": target, "rmse": rmse, "mae": mae}


# 날짜 기반 서브샘플링 — stride일 간격으로 날짜 추출, 해당 날짜의 모든 키워드 유지
def _subsample_by_date(df, stride):
    # stride일 간격으로 날짜 추출, 해당 날짜의 모든 키워드 유지
    if stride <= 1:
        return df
    dates = np.sort(df[DATE_COL].unique())
    keep_dates = dates[::stride]
    return df[df[DATE_COL].isin(keep_dates)]


# 데이터 분할 — 시계열 기반 train/valid/test, gap으로 누수 방지
def split_data(df: pd.DataFrame, train_stride: int = 1):
    if DATE_COL not in df.columns:
        print("날짜 컬럼 없음 → 랜덤 분할")
        train, tmp  = train_test_split(df, test_size=1 - TRAIN_RATIO, random_state=42)
        val_ratio   = VALID_RATIO / (1 - TRAIN_RATIO)
        valid, test = train_test_split(tmp, test_size=1 - val_ratio, random_state=42)
        return train, valid, test

    print(f"시계열 분할 (gap={GAP_DAYS}일)")
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    dates = np.sort(df[DATE_COL].unique())
    n = len(dates)
    train_end = dates[int(n * TRAIN_RATIO)]
    valid_start = train_end + pd.Timedelta(days=GAP_DAYS)

    remaining = dates[dates >= valid_start]
    if len(remaining) == 0:
        raise ValueError("gap 적용 후 valid/test 데이터 없음")
    valid_end = remaining[int(len(remaining) * 0.5)]
    test_start = valid_end + pd.Timedelta(days=GAP_DAYS)

    train = df[df[DATE_COL] <= train_end]
    valid = df[(df[DATE_COL] >= valid_start) & (df[DATE_COL] <= valid_end)]
    test  = df[df[DATE_COL] >= test_start]

    # train/valid만 서브샘플링, test는 stride=1 유지
    if train_stride > 1:
        train = _subsample_by_date(train, train_stride)
        valid = _subsample_by_date(valid, train_stride)

    print(f"  train: ~{train[DATE_COL].min().date()} ~ {train[DATE_COL].max().date()} ({len(train):,})")
    print(f"  gap: {GAP_DAYS}일")
    print(f"  valid: ~{valid[DATE_COL].min().date()} ~ {valid[DATE_COL].max().date()} ({len(valid):,})")
    print(f"  gap: {GAP_DAYS}일")
    print(f"  test:  ~{test[DATE_COL].min().date()} ~ {test[DATE_COL].max().date()} ({len(test):,})")
    if train_stride > 1:
        print(f"  train/valid stride: {train_stride}일 (test는 stride=1)")
    return train, valid, test


# Optuna TPE 튜닝 — SEARCH_SPACE 기반 하이퍼파라미터 탐색, best params JSON 저장
def tune_params(
    X_train, y_train,
    X_valid, y_valid,
    target: str,
    n_trials: int,
    fixed_params: dict,
    dirs: dict,
    w_train=None,
) -> dict:
    print(f"\nOptuna TPE 튜닝 시작: {target}  (trials={n_trials})")

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed_params}
        for name, spec in SEARCH_SPACE.items():
            if spec[0] == "float":
                log = spec[3] if len(spec) > 3 else False
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=log)
            elif spec[0] == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])

        params["n_estimators"] = 3000
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        return rmse_score(y_valid, model.predict(X_valid))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"  best valid RMSE: {study.best_value:.4f}")
    print(f"  best params: {best}")

    out = {"target": target, "best_rmse": study.best_value, "best_params": best}
    path = os.path.join(dirs["metrics"], f"best_params_{target}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return {**fixed_params, **best}


# LightGBM 학습 — early stopping 적용, 최적 트리 수 출력
def train_lgbm(
    X_train, y_train,
    X_valid, y_valid,
    target: str,
    params: dict,
    w_train=None,
) -> lgb.LGBMModel:

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    print(f"  최적 트리 수: {model.best_iteration_}  ({target})")
    return model


# SHAP 분석 — feature importance 시각화(bar/beeswarm) 및 CSV 저장
def shap_analysis(model: lgb.LGBMModel, X: pd.DataFrame, target: str, dirs: dict):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # multiclass: list of arrays → 클래스별 평균 절대값
    if isinstance(shap_values, list):
        shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        shap_abs = np.abs(shap_values)

    plt.figure()
    shap.summary_plot(shap_abs, X, plot_type="bar", show=False)
    plt.title(f"SHAP Importance (bar) — {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["figures"], f"shap_bar_{target}.png"), dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_abs, X, show=False)
    plt.title(f"SHAP Beeswarm — {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["figures"], f"shap_beeswarm_{target}.png"), dpi=150)
    plt.close()

    imp_df = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": shap_abs.mean(0)})
        .sort_values("mean_abs_shap", ascending=False)
    )
    imp_df.to_csv(os.path.join(dirs["metrics"], f"shap_importance_{target}.csv"), index=False)


# 모델 파이프라인 — 튜닝 → 학습 → 평가 → SHAP → 모델 저장
def run_model(
    X_train, y_train,
    X_valid, y_valid,
    X_test,  y_test,
    target: str,
    base_params: dict,
    do_tune: bool,
    n_trials: int,
    dirs: dict,
    w_train=None,
) -> list[dict]:

    print(f"\n{'='*55}")
    print(f"  {target}")
    print(f"{'='*55}")

    if do_tune:
        params = tune_params(
            X_train, y_train, X_valid, y_valid,
            target, n_trials, base_params, dirs, w_train,
        )
    else:
        params = base_params

    model = train_lgbm(X_train, y_train, X_valid, y_valid, target, params, w_train)

    results = []
    for split, X, y in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test",  X_test,  y_test),
    ]:
        results.append(evaluate(y, model.predict(X), split, target))

    shap_analysis(model, X_test, target, dirs)
    model.booster_.save_model(os.path.join(dirs["models"], f"lgbm_{target}.txt"))

    return results


# 메인 — CSV 로드 → 분할 → 타겟별 학습 실행
def main(csv_path: str, target: str, do_tune: bool, n_trials: int, train_stride: int = 3):
    dirs = _make_run_dir(target)
    print(f"output → {os.path.dirname(dirs['figures'])}")
    print(f"target: {target}")

    df = load_csv(csv_path)
    print(f"shape: {df.shape}")

    train_df, valid_df, test_df = split_data(df, train_stride=train_stride)
    print(f"train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")
    print(f"features: {len(FEAT_COLS)}")

    X_train = train_df[FEAT_COLS]
    X_valid = valid_df[FEAT_COLS]
    X_test  = test_df[FEAT_COLS]

    # 타겟별 전처리
    cfg = TARGET_CONFIG.get(target, {"log": False})
    use_log = cfg.get("log", False)

    base_params = {**DEFAULT_PARAMS}
    if use_log:
        y_train = np.log1p(train_df[target])
        y_valid = np.log1p(valid_df[target])
        y_test  = np.log1p(test_df[target])
        w_train = None
        print("log1p 변환 적용")
    else:
        y_train = train_df[target]
        y_valid = valid_df[target]
        y_test  = test_df[target]
        w_train = None

    results = run_model(
        X_train, y_train,
        X_valid, y_valid,
        X_test,  y_test,
        target     = target,
        base_params= base_params,
        do_tune    = do_tune,
        n_trials   = n_trials,
        dirs       = dirs,
        w_train    = w_train,
    )

    # log 변환 사용 시 원래 스케일 복원 평가
    if use_log:
        model_path = os.path.join(dirs["models"], f"lgbm_{target}.txt")
        if os.path.exists(model_path):
            _model = lgb.Booster(model_file=model_path)
            print(f"\n  [원래 스케일 복원 평가]")
            for split, X, y_orig in [
                ("train", X_train, train_df[target]),
                ("valid", X_valid, valid_df[target]),
                ("test",  X_test,  test_df[target]),
            ]:
                pred_orig = np.expm1(_model.predict(X))
                evaluate(y_orig, pred_orig, split, f"{target}_restored")

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(dirs["metrics"], "evaluation_results.csv"), index=False)

    print(f"\n{'='*55}")
    print(res_df.to_string(index=False))
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default=os.path.join(str(ROOT), "data", "processed", "dataset.csv"),
        help="피처 + 라벨 CSV 경로",
    )
    parser.add_argument(
        "--target", type=str, default="sustainability_10d",
        help="학습 타겟 (intensity/buzz_composite/growth/sustainability/crash + _5d/_10d/_15d, 또는 all)",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Optuna TPE 하이퍼파라미터 튜닝 활성화",
    )
    parser.add_argument(
        "--n_trials", type=int, default=25,
        help="Optuna trial 횟수 (기본값: 25, early stopping 적용)",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="GPU 학습 활성화 (CUDA 필요)",
    )
    parser.add_argument(
        "--train-stride", type=int, default=3,
        help="train/valid 서브샘플링 간격 (test는 항상 stride=1, 기본값: 3)",
    )
    args = parser.parse_args()
    if args.gpu:
        DEFAULT_PARAMS["device"] = "cuda"
        print("GPU mode enabled")

    ALL_TARGETS = [
        "intensity_5d", "intensity_10d", "intensity_15d",
        "buzz_composite_5d", "buzz_composite_10d", "buzz_composite_15d",
        "growth_5d", "growth_10d", "growth_15d",
        "sustainability_5d", "sustainability_10d", "sustainability_15d",
        "crash_5d", "crash_10d", "crash_15d",
    ]
    targets = ALL_TARGETS if args.target == "all" else [args.target]
    for t in targets:
        main(args.data, t, args.tune, args.n_trials, args.train_stride)
