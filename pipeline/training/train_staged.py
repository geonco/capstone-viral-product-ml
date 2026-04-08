"""
2단계 peak_time 예측 모델 (분류 → 구간별 회귀)

배경:
단일 회귀(regression_l1)는 median을 찍는 구조라 extreme(빠름/느림)을 복구하지 못함.
구간 분류 → 구간별 회귀 파이프라인으로 구조 자체를 교체.

파이프라인:
  1. LGBMClassifier → 빠름(0) / 중간(1) / 느림(2) 구간 예측
  2. 구간별 LGBMRegressor × 3 → 해당 구간 내 실수 예측
  3. inference: clf 예측 구간 → 해당 regressor 라우팅

구간 기준 (BUCKET_THRESHOLDS):
  - 빠름: peak_time <= 3일
  - 중간: 3일 < peak_time <= 8일
  - 느림: peak_time > 8일

샘플 가중치:
  - extreme(빠름/느림) 구간에 weight=2.0 부여
  - 중간 구간은 weight=1.0

사용법:
    # 기본 실행
    python pipeline/training/train_staged.py --data data/processed/dataset_v2.csv

    # Optuna 튜닝 포함
    python pipeline/training/train_staged.py --data data/processed/dataset_v2.csv --tune --n_trials 50

    # 분할 파일 사용
    python pipeline/training/train_staged.py --use_splits --tune --n_trials 50

출력 구조:
    outputs/MMDD_사용자명_N/
    ├── figures/     # SHAP 그래프 (구간별 + classifier)
    ├── metrics/     # 평가 지표, 파라미터, SHAP 중요도
    └── models/      # lgbm_clf.txt, lgbm_reg_fast.txt, lgbm_reg_mid.txt, lgbm_reg_slow.txt
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
import optuna
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))
from feature_engineering.feature_config import FEATURE_COLS
from data_loader import load_csv
from run_id import make_run_id, DIR_OUTPUTS, write_summary

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── 상수 ──────────────────────────────────────────────────────────────────────
TARGET_TIMING = "peak_time"
DATE_COL      = "date"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15

# 구간 경계: (fast_upper, mid_upper]
# fast:  peak_time <= 3
# mid:   3 < peak_time <= 9
# slow:  peak_time > 9
BUCKET_FAST_MAX = 3
BUCKET_MID_MAX  = 9

BUCKET_NAMES = {0: "fast", 1: "mid", 2: "slow"}

# extreme 구간(빠름/느림) 샘플 가중치
EXTREME_WEIGHT = 2.0

DEFAULT_CLF_PARAMS = {
    "boosting_type"    : "gbdt",
    "objective"        : "multiclass",
    "num_class"        : 3,
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
    "metric"           : "multi_logloss",
}

DEFAULT_REG_PARAMS = {
    "boosting_type"    : "gbdt",
    "objective"        : "regression",   # L2: variance 반영 (L1이 아님)
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

SEARCH_SPACE = {
    "learning_rate"    : ("float", 1e-3, 0.3,   True),
    "num_leaves"       : ("int",   16,   256),
    "max_depth"        : ("int",   3,    12),
    "min_child_samples": ("int",   5,    100),
    "subsample"        : ("float", 0.5,  1.0,   False),
    "colsample_bytree" : ("float", 0.5,  1.0,   False),
    "reg_alpha"        : ("float", 1e-4, 10.0,  True),
    "reg_lambda"       : ("float", 1e-4, 10.0,  True),
}


# ── 유틸 ──────────────────────────────────────────────────────────────────────
def rmse_score(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_bucket(y: pd.Series | np.ndarray) -> np.ndarray:
    """peak_time → 구간 레이블 (0: 빠름, 1: 중간, 2: 느림)"""
    y = np.asarray(y)
    buckets = np.where(
        y <= BUCKET_FAST_MAX, 0,
        np.where(y <= BUCKET_MID_MAX, 1, 2)
    )
    return buckets


def make_sample_weight(bucket: np.ndarray) -> np.ndarray:
    """extreme 구간(0, 2)에 가중치 EXTREME_WEIGHT 부여"""
    return np.where((bucket == 0) | (bucket == 2), EXTREME_WEIGHT, 1.0)


def add_slope_features(df: pd.DataFrame) -> pd.DataFrame:
    """기존 피처 조합으로 타이밍 구분 피처 8개 추가

    [기울기/가속도 그룹 - 3개]
    slope_ratio_3_30  : search_slope_3d / (|search_slope_30d| + ε)
                        단기 기울기가 장기 대비 얼마나 가파른가.
                        빠른 바이럴은 값이 크고, 느린 바이럴은 작거나 음수.
    ma_ratio_3_30     : search_ma_3d / (search_ma_30d + ε)
                        단기 이동평균 / 장기 이동평균. 1보다 크면 최근 급상승.
    accel_sign_match  : search_accel_short * search_accel_long > 0 → 1, 아니면 0
                        단기·장기 가속도 방향이 일치하면 트렌드 지속성이 높음.

    [타이밍 직접 신호 그룹 - 5개]
    search_early_conc : search_ma_3d / (search_ma_7d + ε)
                        초기 3일이 7일 대비 얼마나 집중됐는가.
                        빠른 바이럴은 초반에 트래픽이 몰려 값이 큼.
    click_early_conc  : click_ma_3d / (click_ma_7d + ε)
                        클릭 신호의 초기 집중도.
    mention_early_conc: mention_ma_3d / (mention_ma_7d + ε)
                        언급 신호의 초기 집중도.
    days_since_max_mean: (days_since_search_max_7d + days_since_click_max_7d
                          + days_since_mention_max_7d) / 3
                        3개 신호 기준 피크 이후 평균 경과일.
                        값이 작으면 피크가 최근 → 빠른 바이럴,
                        값이 크면 피크가 오래 전 → 느린 바이럴.
    signal_sync       : std(days_since_search_max_7d, days_since_click_max_7d,
                           days_since_mention_max_7d)
                        3개 신호의 피크 타이밍 동기화 정도.
                        값이 작으면 동시에 피크 → 명확한 바이럴 패턴.
    """
    eps = 1e-6
    df = df.copy()

    # 기울기/가속도 그룹
    df["slope_ratio_3_30"] = df["search_slope_3d"] / (df["search_slope_30d"].abs() + eps)
    df["ma_ratio_3_30"]    = df["search_ma_3d"] / (df["search_ma_30d"] + eps)
    df["accel_sign_match"] = ((df["search_accel_short"] * df["search_accel_long"]) > 0).astype(float)

    # 타이밍 직접 신호 그룹
    df["search_early_conc"]  = df["search_ma_3d"]  / (df["search_ma_7d"]  + eps)
    df["click_early_conc"]   = df["click_ma_3d"]   / (df["click_ma_7d"]   + eps)
    df["mention_early_conc"] = df["mention_ma_3d"] / (df["mention_ma_7d"] + eps)

    days_cols = ["days_since_search_max_7d", "days_since_click_max_7d", "days_since_mention_max_7d"]
    df["days_since_max_mean"] = df[days_cols].mean(axis=1)
    df["signal_sync"]         = df[days_cols].std(axis=1)

    return df


# 기존 FEATURE_COLS에 새 피처 추가 (staged 모델 전용, feature_config.py 미수정)
STAGED_FEATURE_COLS = FEATURE_COLS + [
    "slope_ratio_3_30", "ma_ratio_3_30", "accel_sign_match",
    "search_early_conc", "click_early_conc", "mention_early_conc",
    "days_since_max_mean", "signal_sync",
]


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


def print_bucket_dist(label: str, y: np.ndarray, buckets: np.ndarray):
    """구간 분포 출력"""
    total = len(buckets)
    print(f"\n  [{label}] 구간 분포 (total={total})")
    for b, name in BUCKET_NAMES.items():
        mask = buckets == b
        cnt  = mask.sum()
        if cnt > 0:
            print(f"    {name:4s} (bucket={b}): {cnt:4d}개  ({cnt/total*100:.1f}%)  "
                  f"peak_time 범위 [{y[mask].min():.0f}, {y[mask].max():.0f}]")
        else:
            print(f"    {name:4s} (bucket={b}): 0개  (0.0%)")


# ── Optuna 튜닝 ───────────────────────────────────────────────────────────────
def _suggest_params(trial: optuna.Trial, fixed: dict) -> dict:
    params = {**fixed}
    for name, spec in SEARCH_SPACE.items():
        if spec[0] == "float":
            log = spec[3] if len(spec) > 3 else False
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=log)
        elif spec[0] == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
    params["n_estimators"] = 1000
    return params


def tune_classifier(
    X_train, y_train, w_train,
    X_valid, y_valid,
    n_trials: int,
    dir_metrics: str,
) -> dict:
    print(f"\nOptuna TPE 튜닝: classifier  (trials={n_trials})")

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, DEFAULT_CLF_PARAMS)
        model  = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        preds = model.predict(X_valid)
        return 1.0 - accuracy_score(y_valid, preds)   # minimize error rate

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"  최적 valid error: {study.best_value:.4f}")

    out  = {"target": "classifier", "best_error": study.best_value, "best_params": best}
    path = os.path.join(dir_metrics, "best_params_clf.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  저장: {path}")

    return {**DEFAULT_CLF_PARAMS, **best}


def tune_regressor(
    X_train, y_train, w_train,
    X_valid, y_valid,
    bucket_name: str,
    n_trials: int,
    dir_metrics: str,
) -> dict:
    print(f"\nOptuna TPE 튜닝: regressor_{bucket_name}  (trials={n_trials})")

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, DEFAULT_REG_PARAMS)
        model  = lgb.LGBMRegressor(**params)
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
    print(f"  최적 valid RMSE: {study.best_value:.4f}")

    out  = {"target": f"reg_{bucket_name}", "best_rmse": study.best_value, "best_params": best}
    path = os.path.join(dir_metrics, f"best_params_reg_{bucket_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  저장: {path}")

    return {**DEFAULT_REG_PARAMS, **best}


# ── 학습 ──────────────────────────────────────────────────────────────────────
def train_classifier(
    X_train, y_train, w_train,
    X_valid, y_valid,
    params: dict,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    print(f"  최적 트리 수: {model.best_iteration_}  (classifier)")
    return model


def train_regressor(
    X_train, y_train, w_train,
    X_valid, y_valid,
    bucket_name: str,
    params: dict,
) -> lgb.LGBMRegressor:
    if len(X_train) == 0:
        print(f"  경고: {bucket_name} 구간 학습 데이터 없음 → 건너뜀")
        return None

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
    print(f"  최적 트리 수: {model.best_iteration_}  (reg_{bucket_name})")
    return model


# ── 추론 ──────────────────────────────────────────────────────────────────────
def staged_predict(
    clf: lgb.LGBMClassifier,
    regressors: dict,   # {0: model_fast, 1: model_mid, 2: model_slow}
    X: pd.DataFrame,
    fallback_values: dict,  # 모델 없을 때 bucket별 median
) -> np.ndarray:
    """soft routing: clf 확률값 가중합으로 각 구간 regressor 혼합

    hard routing(argmax)과 달리 clf가 틀려도 확률 분포가 정보를 유지하므로
    overall MAE 감소 효과가 있음.
    """
    proba = clf.predict_proba(X)   # (n, 3) — [P(fast), P(mid), P(slow)]
    preds = np.zeros(len(X))

    for b, name in BUCKET_NAMES.items():
        model = regressors.get(b)
        if model is None:
            reg_pred = np.full(len(X), fallback_values.get(b, 5.0))
        else:
            reg_pred = model.predict(X)
        preds += proba[:, b] * reg_pred

    return preds


# ── SHAP 분석 ─────────────────────────────────────────────────────────────────
def shap_analysis(model, X: pd.DataFrame, label: str, dir_figures: str, dir_metrics: str):
    if model is None or len(X) == 0:
        return
    print(f"\nSHAP 분석: {label}")
    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # multiclass인 경우 shap_values는 list → 클래스별 처리
        if isinstance(shap_values, list):
            for i, sv in enumerate(shap_values):
                _save_shap_plots(sv, X, f"{label}_class{i}", dir_figures, dir_metrics)
        else:
            _save_shap_plots(shap_values, X, label, dir_figures, dir_metrics)
    except Exception as e:
        print(f"  SHAP 분석 실패 ({label}): {e}")


def _save_shap_plots(shap_values, X: pd.DataFrame, label: str, dir_figures: str, dir_metrics: str):
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"SHAP Importance (bar) — {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_figures, f"shap_bar_{label}.png"), dpi=150)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Beeswarm — {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(dir_figures, f"shap_beeswarm_{label}.png"), dpi=150)
    plt.close()

    imp_df = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(shap_values).mean(0)})
        .sort_values("mean_abs_shap", ascending=False)
    )
    imp_df.to_csv(os.path.join(dir_metrics, f"shap_importance_{label}.csv"), index=False)


# ── 평가 ──────────────────────────────────────────────────────────────────────
def evaluate_staged(
    clf: lgb.LGBMClassifier,
    regressors: dict,
    fallback_values: dict,
    X: pd.DataFrame,
    y_true: np.ndarray,
    split: str,
) -> list[dict]:
    results = []

    # 전체 예측
    y_pred = staged_predict(clf, regressors, X, fallback_values)
    rmse   = rmse_score(y_true, y_pred)
    mae    = float(mean_absolute_error(y_true, y_pred))
    print(f"  [{split:5s}] overall  RMSE={rmse:.4f}  MAE={mae:.4f}")
    results.append({"split": split, "bucket": "overall", "rmse": rmse, "mae": mae, "n": len(y_true)})

    # classifier 정확도
    true_buckets = make_bucket(y_true)
    pred_buckets = clf.predict(X)
    acc = accuracy_score(true_buckets, pred_buckets)
    print(f"  [{split:5s}] clf accuracy={acc:.4f}")
    results.append({"split": split, "bucket": "clf_accuracy", "rmse": None, "mae": acc, "n": len(y_true)})

    # 구간별 평가 (true bucket 기준)
    for b, name in BUCKET_NAMES.items():
        mask = true_buckets == b
        if mask.sum() == 0:
            continue
        model = regressors.get(b)
        if model is None:
            continue
        yb_true = y_true[mask]
        yb_pred = model.predict(X[mask])
        rmse_b  = rmse_score(yb_true, yb_pred)
        mae_b   = float(mean_absolute_error(yb_true, yb_pred))
        print(f"  [{split:5s}] {name:4s}  RMSE={rmse_b:.4f}  MAE={mae_b:.4f}  (n={mask.sum()})")
        results.append({"split": split, "bucket": name, "rmse": rmse_b, "mae": mae_b, "n": int(mask.sum())})

    return results


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main(
    csv_path: str,
    use_splits: bool,
    do_tune: bool,
    n_trials: int,
    user: str | None = None,
):
    # 실험 폴더
    experiment_name = make_run_id(DIR_OUTPUTS, user)
    exp_dir         = os.path.join(DIR_OUTPUTS, experiment_name)
    dir_figures     = os.path.join(exp_dir, "figures")
    dir_metrics     = os.path.join(exp_dir, "metrics")
    dir_models      = os.path.join(exp_dir, "models")
    for d in [dir_figures, dir_metrics, dir_models]:
        os.makedirs(d, exist_ok=True)

    print(f"실험 폴더 생성: {experiment_name}")

    # 데이터 로드
    if use_splits:
        print("분할된 데이터 사용: data_processed/splits/")
        train_df = load_csv("data_processed/splits/train.csv")
        valid_df = load_csv("data_processed/splits/valid.csv")
        test_df  = load_csv("data_processed/splits/test.csv")
    else:
        df = load_csv(csv_path)
        print(f"데이터 로드: {df.shape}  ({csv_path})")
        train_df, valid_df, test_df = split_data(df)

    print(f"train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")

    # 파생 피처 추가 (slope_ratio_3_30, ma_ratio_3_30, accel_sign_match)
    train_df = add_slope_features(train_df)
    valid_df = add_slope_features(valid_df)
    test_df  = add_slope_features(test_df)
    print(f"feature 수: {len(STAGED_FEATURE_COLS)}  (base {len(FEATURE_COLS)} + 추가 3)")

    X_train = train_df[STAGED_FEATURE_COLS]
    X_valid = valid_df[STAGED_FEATURE_COLS]
    X_test  = test_df[STAGED_FEATURE_COLS]

    y_train = train_df[TARGET_TIMING].values
    y_valid = valid_df[TARGET_TIMING].values
    y_test  = test_df[TARGET_TIMING].values

    # 구간 레이블 + 샘플 가중치
    b_train = make_bucket(y_train)
    b_valid = make_bucket(y_valid)
    b_test  = make_bucket(y_test)

    w_train = make_sample_weight(b_train)

    print_bucket_dist("train", y_train, b_train)
    print_bucket_dist("valid", y_valid, b_valid)
    print_bucket_dist("test",  y_test,  b_test)

    # ── STEP 1: classifier ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  STEP 1: Classifier (fast / mid / slow)")
    print(f"{'='*55}")

    if do_tune:
        clf_params = tune_classifier(
            X_train, b_train, w_train,
            X_valid, b_valid,
            n_trials, dir_metrics,
        )
    else:
        saved_clf = os.path.join(dir_metrics, "best_params_clf.json")
        if os.path.isfile(saved_clf):
            with open(saved_clf, encoding="utf-8") as f:
                saved = json.load(f)
            clf_params = {**DEFAULT_CLF_PARAMS, **saved["best_params"]}
            print(f"  저장된 clf 파라미터 로드 (best_error={saved['best_error']:.4f})")
        else:
            clf_params = DEFAULT_CLF_PARAMS

    clf = train_classifier(X_train, b_train, w_train, X_valid, b_valid, clf_params)
    clf.booster_.save_model(os.path.join(dir_models, "lgbm_clf.txt"))

    # valid 분류 리포트
    print("\n  [valid] 분류 리포트")
    print(classification_report(b_valid, clf.predict(X_valid),
                                 target_names=["fast", "mid", "slow"]))

    shap_analysis(clf, X_test, "clf", dir_figures, dir_metrics)

    # ── STEP 2: 구간별 regressor ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  STEP 2: Per-bucket Regressors")
    print(f"{'='*55}")

    regressors     = {}
    fallback_values = {}

    for b, name in BUCKET_NAMES.items():
        mask_tr = b_train == b
        mask_va = b_valid == b

        fallback_values[b] = float(np.median(y_train[mask_tr])) if mask_tr.sum() > 0 else 5.0

        print(f"\n--- {name} (bucket={b}) ---")
        print(f"  학습 샘플: {mask_tr.sum()}개 / valid 샘플: {mask_va.sum()}개")

        if mask_tr.sum() < 10:
            print(f"  경고: 학습 샘플 부족 ({mask_tr.sum()}개) → 건너뜀")
            regressors[b] = None
            continue

        if mask_va.sum() == 0:
            print(f"  경고: valid 샘플 없음 → 기본 파라미터 사용")
            reg_params = DEFAULT_REG_PARAMS
        elif do_tune:
            reg_params = tune_regressor(
                X_train[mask_tr], y_train[mask_tr], w_train[mask_tr],
                X_valid[mask_va], y_valid[mask_va],
                name, n_trials, dir_metrics,
            )
        else:
            saved_reg = os.path.join(dir_metrics, f"best_params_reg_{name}.json")
            if os.path.isfile(saved_reg):
                with open(saved_reg, encoding="utf-8") as f:
                    saved = json.load(f)
                reg_params = {**DEFAULT_REG_PARAMS, **saved["best_params"]}
                print(f"  저장된 reg_{name} 파라미터 로드 (best_rmse={saved['best_rmse']:.4f})")
            else:
                reg_params = DEFAULT_REG_PARAMS

        # valid 샘플 없으면 train 일부를 valid 대용으로 사용
        if mask_va.sum() == 0:
            xv_b = X_train[mask_tr].iloc[:max(1, mask_tr.sum() // 5)]
            yv_b = y_train[mask_tr][:max(1, mask_tr.sum() // 5)]
            wv_b = w_train[mask_tr][:max(1, mask_tr.sum() // 5)]
        else:
            xv_b = X_valid[mask_va]
            yv_b = y_valid[mask_va]
            wv_b = make_sample_weight(b_valid[mask_va])

        model = train_regressor(
            X_train[mask_tr], y_train[mask_tr], w_train[mask_tr],
            xv_b, yv_b,
            name, reg_params,
        )
        regressors[b] = model

        if model is not None:
            model.booster_.save_model(os.path.join(dir_models, f"lgbm_reg_{name}.txt"))
            shap_analysis(model, X_test[b_test == b] if (b_test == b).sum() > 0 else X_test.iloc[:1],
                          f"reg_{name}", dir_figures, dir_metrics)

    # ── STEP 3: 전체 평가 ────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  STEP 3: 전체 평가")
    print(f"{'='*55}")

    all_results = []
    for split, X, y in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test",  X_test,  y_test),
    ]:
        all_results += evaluate_staged(clf, regressors, fallback_values, X, y, split)

    # 결과 저장
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(os.path.join(dir_metrics, "evaluation_results.csv"), index=False)

    print(f"\n{'='*55}")
    print("최종 평가 결과")
    print("="*55)
    print(res_df.to_string(index=False))
    print(f"\n모든 결과 저장 완료 → outputs/{experiment_name}/")

    # ── 실험 요약 ─────────────────────────────────────────────────────────────
    now_str    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_desc  = "data_processed/splits/" if use_splits else csv_path
    split_desc = "외부 분할 파일" if use_splits else f"자동 분할 (train {int(TRAIN_RATIO*100)}% / valid {int(VALID_RATIO*100)}%)"
    tune_desc  = f"Optuna TPE ({n_trials} trials)" if do_tune else "비활성화"

    rows = []
    for r in all_results:
        rmse_str = f"{r['rmse']:.4f}" if r["rmse"] is not None else "-"
        rows.append(f"| {r['split']} | {r['bucket']} | {rmse_str} | {r['mae']:.4f} | {r['n']} |")
    result_table = "\n".join(rows)

    markdown = f"""# 실험 요약 — {experiment_name}

## 기본 정보
- 유형: 2단계 모델 (분류 → 구간별 회귀)
- 일시: {now_str}
- 데이터: {data_desc}
- 분할: {split_desc}
- 튜닝: {tune_desc}
- feature 수: {len(STAGED_FEATURE_COLS)}  (base {len(FEATURE_COLS)} + 파생 {len(STAGED_FEATURE_COLS)-len(FEATURE_COLS)})
- 구간 기준: fast(≤{BUCKET_FAST_MAX}일) / mid(≤{BUCKET_MID_MAX}일) / slow(>{BUCKET_MID_MAX}일)
- extreme 샘플 가중치: {EXTREME_WEIGHT}
- 데이터 크기: train {len(train_df)} / valid {len(valid_df)} / test {len(test_df)}

## 평가 결과

| split | bucket | RMSE | MAE / accuracy | n |
|-------|--------|------|----------------|---|
{result_table}

## 생성 파일
- metrics/evaluation_results.csv
- metrics/best_params_clf.json (튜닝 시)
- metrics/best_params_reg_{{bucket}}.json (튜닝 시)
- metrics/shap_importance_{{label}}.csv
- figures/shap_{{bar,beeswarm}}_{{label}}.png
- models/lgbm_clf.txt
- models/lgbm_reg_{{fast,mid,slow}}.txt
"""
    write_summary(exp_dir, markdown)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="data/processed/dataset_v2.csv",
        help="feature + target 컬럼이 포함된 CSV 경로",
    )
    parser.add_argument(
        "--use_splits", action="store_true",
        help="data_processed/splits/ 폴더의 분할 파일 사용",
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
        "--user", type=str, default=None,
        help="실험자 이름 (최초 1회 지정하면 outputs/.username에 저장됨)",
    )
    args = parser.parse_args()
    main(args.data, args.use_splits, args.tune, args.n_trials, args.user)
