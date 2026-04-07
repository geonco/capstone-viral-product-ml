# LightGBM 바이럴 예측 모델 (Optuna TPE 하이퍼파라미터 튜닝 포함)
# - virality_score : z-score 기반 바이럴 강도 (60일 lookback, 14일 forward)
# - peak_time      : argmax(search[t:t+14]), 0~13

import argparse
import json
import os
import sys
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))
from config import FEAT_COLS
from data_loader import load_csv

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 상수 ──────────────────────────────────────────────────────────────────────
TARGET_VIRALITY = "virality_score"
DATE_COL        = "date"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15

DEFAULT_PARAMS = {
    "boosting_type"    : "gbdt",
    "n_estimators"     : 1000,
    "learning_rate"    : 0.05,
    "num_leaves"       : 64,
    "max_depth"        : -1,
    "min_child_samples": 20,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "reg_alpha"        : 0.1,
    "reg_lambda"       : 0.1,
    "random_state"     : 42,
    "n_jobs"           : -1,
    "verbose"          : -1,
}

# Optuna TPE 탐색 범위
SEARCH_SPACE = {
    "learning_rate"    : ("float", 0.01, 0.15,  True),
    "num_leaves"       : ("int",   15,   127),
    "max_depth"        : ("int",   4,    10),
    "min_child_samples": ("int",   50,   500),
    "subsample"        : ("float", 0.6,  0.95,  False),
    "colsample_bytree" : ("float", 0.4,  0.9,   False),
    "reg_alpha"        : ("float", 1e-3, 10.0,  True),
    "reg_lambda"       : ("float", 1e-3, 10.0,  True),
}

_ROOT = os.path.join(os.path.dirname(__file__), "..")

def _make_run_dir():
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = os.path.join(_ROOT, "outputs", f"run_{stamp}")
    dirs = {
        "figures": os.path.join(run, "figures"),
        "metrics": os.path.join(run, "metrics"),
        "models":  os.path.join(run, "models"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


# ── 지표 ──────────────────────────────────────────────────────────────────────
def rmse_score(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred, split: str, target: str) -> dict:
    rmse = rmse_score(y_true, y_pred)
    mae  = float(mean_absolute_error(y_true, y_pred))
    print(f"  [{split:5s}] {target}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return {"split": split, "target": target, "rmse": rmse, "mae": mae}


# ── 데이터 분할 ───────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    if DATE_COL in df.columns:
        print(f"'{DATE_COL}' 컬럼 감지 → 시계열 순서 기반 분할")
        df = df.sort_values(DATE_COL).reset_index(drop=True)
        n  = len(df)
        t1 = int(n * TRAIN_RATIO)
        t2 = int(n * (TRAIN_RATIO + VALID_RATIO))
        return df.iloc[:t1], df.iloc[t1:t2], df.iloc[t2:]
    else:
        print("날짜 컬럼 없음 → 랜덤 분할")
        train, tmp  = train_test_split(df, test_size=1 - TRAIN_RATIO, random_state=42)
        val_ratio   = VALID_RATIO / (1 - TRAIN_RATIO)
        valid, test = train_test_split(tmp, test_size=1 - val_ratio, random_state=42)
        return train, valid, test


# ── Optuna TPE 튜닝 ───────────────────────────────────────────────────────────
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
                lgb.early_stopping(stopping_rounds=50, verbose=False),
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


# ── LightGBM 학습 ─────────────────────────────────────────────────────────────
def train_lgbm(
    X_train, y_train,
    X_valid, y_valid,
    target: str,
    params: dict,
    w_train=None,
) -> lgb.LGBMRegressor:

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    print(f"  최적 트리 수: {model.best_iteration_}  ({target})")
    return model


# ── SHAP 분석 ─────────────────────────────────────────────────────────────────
def shap_analysis(model: lgb.LGBMRegressor, X: pd.DataFrame, target: str, dirs: dict):
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"SHAP Importance (bar) — {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["figures"], f"shap_bar_{target}.png"), dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Beeswarm — {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["figures"], f"shap_beeswarm_{target}.png"), dpi=150)
    plt.close()

    imp_df = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(shap_values).mean(0)})
        .sort_values("mean_abs_shap", ascending=False)
    )
    imp_df.to_csv(os.path.join(dirs["metrics"], f"shap_importance_{target}.csv"), index=False)


# ── 모델 파이프라인 ────────────────────────────────────────────────────────────
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


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main(csv_path: str, target: str, do_tune: bool, n_trials: int):
    dirs = _make_run_dir()
    print(f"output → {os.path.dirname(dirs['figures'])}")
    print(f"target: {target}")

    df = load_csv(csv_path)
    print(f"shape: {df.shape}")

    train_df, valid_df, test_df = split_data(df)
    print(f"train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")
    print(f"features: {len(FEAT_COLS)}")

    X_train = train_df[FEAT_COLS]
    X_valid = valid_df[FEAT_COLS]
    X_test  = test_df[FEAT_COLS]

    # 타겟별 전처리
    use_log = target in ("virality_score", "click_weighted_lift", "future_ratio")
    if use_log:
        y_train = np.log1p(train_df[target])
        y_valid = np.log1p(valid_df[target])
        y_test  = np.log1p(test_df[target])
        w_train = 1.0 + np.log1p(train_df[target].values)
        print(f"log1p 변환 적용, sample weight: min={w_train.min():.2f}  max={w_train.max():.2f}")
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
        base_params= {**DEFAULT_PARAMS},
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
        "--data", type=str, default="data/processed/dataset_20260403.csv",
        help="216피처 + 라벨 CSV 경로",
    )
    parser.add_argument(
        "--target", type=str, default="virality_score",
        help="학습 타겟 (virality_score, viral_percentile, concordance, sustained_breakout, trajectory_class, click_weighted_lift)",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Optuna TPE 하이퍼파라미터 튜닝 활성화",
    )
    parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Optuna trial 횟수 (기본값: 50)",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="GPU 학습 활성화 (CUDA 필요)",
    )
    args = parser.parse_args()
    if args.gpu:
        DEFAULT_PARAMS["device"] = "gpu"
        DEFAULT_PARAMS["gpu_use_dp"] = False
        print("GPU mode enabled")

    ALL_TARGETS = [
        "virality_score", "viral_percentile", "concordance",
        "sustained_breakout", "trajectory_class", "click_weighted_lift",
    ]
    targets = ALL_TARGETS if args.target == "all" else [args.target]
    for t in targets:
        main(args.data, t, args.tune, args.n_trials)
