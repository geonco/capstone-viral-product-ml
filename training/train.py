"""
LightGBM 바이럴 예측 모델  (Optuna TPE 하이퍼파라미터 튜닝 포함)
- virality_score : (future_peak - baseline_90d) / (std_90d + 1e-6)
- peak_timing    : argmax(search_trend[t+1:t+30]) + 1  (1~30일)

사용법:
    # 기본 파라미터로 학습
    python training/train.py --data data_raw/dataset_final.csv

    # Optuna TPE 튜닝 후 최적 파라미터로 학습 (trial 50회)
    python training/train.py --data data_raw/dataset_final.csv --tune --n_trials 50
"""

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

# feature_engineering 모듈 import
sys.path.append(os.path.dirname(__file__))
from feature_engineering.feature_config import FEATURE_COLS
from data_loader import load_csv

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 상수 ──────────────────────────────────────────────────────────────────────
TARGET_VIRALITY = "virality_score"
TARGET_TIMING   = "peak_time"
DATE_COL        = "date"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
# TEST_RATIO  = 0.15

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
    "learning_rate"    : ("float", 1e-3, 0.3,   True),   # (type, low, high, log)
    "num_leaves"       : ("int",   16,   256),
    "max_depth"        : ("int",   3,    12),
    "min_child_samples": ("int",   5,    100),
    "subsample"        : ("float", 0.5,  1.0,   False),
    "colsample_bytree" : ("float", 0.5,  1.0,   False),
    "reg_alpha"        : ("float", 1e-4, 10.0,  True),
    "reg_lambda"       : ("float", 1e-4, 10.0,  True),
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")


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
) -> dict:
    """
    Optuna TPE Sampler로 valid RMSE 최소화 파라미터 탐색.
    이전 trial 결과를 반영해 유망한 파라미터 범위를 점점 좁혀감.
    """
    print(f"\nOptuna TPE 튜닝 시작: {target}  (trials={n_trials})")

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed_params}
        for name, spec in SEARCH_SPACE.items():
            if spec[0] == "float":
                log = spec[3] if len(spec) > 3 else False
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=log)
            elif spec[0] == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])

        params["n_estimators"] = 1000  # early stopping 으로 결정

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
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
    print(f"  최적 valid RMSE : {study.best_value:.4f}")
    print(f"  최적 파라미터   : {best}")

    # 결과 저장
    out = {"target": target, "best_rmse": study.best_value, "best_params": best}
    path = os.path.join(OUTPUT_DIR, f"best_params_{target}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  저장: {path}")

    return {**fixed_params, **best}


# ── LightGBM 학습 ─────────────────────────────────────────────────────────────
def train_lgbm(
    X_train, y_train,
    X_valid, y_valid,
    target: str,
    params: dict,
) -> lgb.LGBMRegressor:

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    print(f"  최적 트리 수: {model.best_iteration_}  ({target})")
    return model


# ── SHAP 분석 ─────────────────────────────────────────────────────────────────
def shap_analysis(model: lgb.LGBMRegressor, X: pd.DataFrame, target: str):
    print(f"\nSHAP 분석: {target}")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # bar plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"SHAP Importance (bar) — {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_bar_{target}.png"), dpi=150)
    plt.close()

    # beeswarm plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Beeswarm — {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_beeswarm_{target}.png"), dpi=150)
    plt.close()

    # 중요도 CSV
    imp_df = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(shap_values).mean(0)})
        .sort_values("mean_abs_shap", ascending=False)
    )
    imp_df.to_csv(os.path.join(OUTPUT_DIR, f"shap_importance_{target}.csv"), index=False)
    print(f"  저장 완료 → {OUTPUT_DIR}/shap_*_{target}.*")


# ── 모델 파이프라인 ────────────────────────────────────────────────────────────
def run_model(
    X_train, y_train,
    X_valid, y_valid,
    X_test,  y_test,
    target: str,
    base_params: dict,
    do_tune: bool,
    n_trials: int,
) -> list[dict]:

    print(f"\n{'='*55}")
    print(f"  {target} 모델")
    print(f"{'='*55}")

    if do_tune:
        params = tune_params(
            X_train, y_train, X_valid, y_valid,
            target, n_trials, base_params,
        )
    else:
        saved_path = os.path.join(OUTPUT_DIR, f"best_params_{target}.json")
        if os.path.isfile(saved_path):
            with open(saved_path, encoding="utf-8") as f:
                saved = json.load(f)
            params = {**base_params, **saved["best_params"]}
            print(f"  저장된 최적 파라미터 로드: {saved_path}  (best_rmse={saved['best_rmse']:.4f})")
        else:
            params = base_params

    model = train_lgbm(X_train, y_train, X_valid, y_valid, target, params)

    results = []
    for split, X, y in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test",  X_test,  y_test),
    ]:
        results.append(evaluate(y, model.predict(X), split, target))

    shap_analysis(model, X_test, target)
    model.booster_.save_model(os.path.join(OUTPUT_DIR, f"lgbm_{target}.txt"))

    return results


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main(csv_path: str, do_tune: bool, n_trials: int):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 데이터 로드 (컬럼 검증 및 재정렬 포함)
    df = load_csv(csv_path)
    print(f"데이터 로드: {df.shape}  ({csv_path})")

    # 2. 분할
    train_df, valid_df, test_df = split_data(df)
    print(f"train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")
    print(f"feature 수: {len(FEATURE_COLS)}")

    X_train = train_df[FEATURE_COLS]
    X_valid = valid_df[FEATURE_COLS]
    X_test  = test_df[FEATURE_COLS]

    # 3. virality_score 모델
    results = run_model(
        X_train, train_df[TARGET_VIRALITY],
        X_valid, valid_df[TARGET_VIRALITY],
        X_test,  test_df[TARGET_VIRALITY],
        target     = TARGET_VIRALITY,
        base_params= {**DEFAULT_PARAMS},
        do_tune    = do_tune,
        n_trials   = n_trials,
    )

    # 4. peak_timing 모델 (MAE 직접 최적화)
    results += run_model(
        X_train, train_df[TARGET_TIMING],
        X_valid, valid_df[TARGET_TIMING],
        X_test,  test_df[TARGET_TIMING],
        target     = TARGET_TIMING,
        base_params= {**DEFAULT_PARAMS, "objective": "regression_l1"},
        do_tune    = do_tune,
        n_trials   = n_trials,
    )

    # 5. 결과 저장
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results.csv"), index=False)

    print(f"\n{'='*55}")
    print("최종 평가 결과")
    print("="*55)
    print(res_df.to_string(index=False))
    print(f"\n모든 결과 저장 완료 → {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="data_processed/features.csv",
        help="feature + target 컬럼이 포함된 CSV 경로",
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Optuna TPE 하이퍼파라미터 튜닝 활성화",
    )
    parser.add_argument(
        "--n_trials", type=int, default=50,
        help="Optuna trial 횟수 (기본값: 50)",
    )
    args = parser.parse_args()
    main(args.data, args.tune, args.n_trials)
