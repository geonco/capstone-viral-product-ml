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
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import (
    ROOT, FEAT_COLS, MASK_REGISTRY, MASK_COMBOS,
    DATE_COL, TRAIN_RATIO, VALID_RATIO, GAP_DAYS,
    TARGET_CONFIG, DEFAULT_PARAMS, SEARCH_SPACE,
)
from pipeline.training.data_loader import load_csv

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def _make_run_dir(target: str = "", combo_label: str = ""):
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{target}" if target else ""
    if combo_label:
        suffix += f"_{combo_label}"
    run = os.path.join(str(ROOT), "outputs", f"run_{stamp}{suffix}")
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


# Hurdle 분류기 파라미터 — base_params를 binary classification용으로 변환
def _make_clf_params(base_params: dict) -> dict:
    params = {**base_params}
    params["objective"] = "binary"
    params["metric"] = "binary_logloss"
    return params


# Hurdle 분류기 튜닝 — F1 최대화 기준 Optuna TPE 탐색
def tune_hurdle_clf(
    X_train, y_train_bin,
    X_valid, y_valid_bin,
    target: str,
    n_trials: int,
    fixed_params: dict,
    dirs: dict,
) -> dict:
    print(f"\nOptuna TPE 튜닝 (분류기): {target}  (trials={n_trials})")
    clf_fixed = _make_clf_params(fixed_params)

    def objective(trial: optuna.Trial) -> float:
        params = {**clf_fixed}
        for name, spec in SEARCH_SPACE.items():
            if spec[0] == "float":
                log = spec[3] if len(spec) > 3 else False
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=log)
            elif spec[0] == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2])

        params["n_estimators"] = 3000
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train_bin,
            eval_set=[(X_valid, y_valid_bin)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
        pred = model.predict(X_valid)
        return f1_score(y_valid_bin, pred)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"  best valid F1: {study.best_value:.4f}")
    print(f"  best params: {best}")

    out = {"target": f"{target}_clf", "best_f1": study.best_value, "best_params": best}
    path = os.path.join(dirs["metrics"], f"best_params_{target}_clf.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return {**clf_fixed, **best}


# Hurdle 분류기 학습 — early stopping 적용, 최적 트리 수 출력
def train_hurdle_clf(
    X_train, y_train_bin,
    X_valid, y_valid_bin,
    target: str,
    params: dict,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train_bin,
        eval_set=[(X_valid, y_valid_bin)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    print(f"  분류기 최적 트리 수: {model.best_iteration_}  ({target})")
    return model


# Hurdle 평가 — 분류 지표(P/R/F1)와 결합 예측의 MAE/RMSE 산출
def evaluate_hurdle(
    y_true, y_pred_bin, y_pred_combined,
    split: str, target: str, threshold: float,
) -> dict:
    y_true_bin = (np.array(y_true) > threshold).astype(int)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec  = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1   = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    mae  = float(mean_absolute_error(y_true, y_pred_combined))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_combined)))

    n_true = int(y_true_bin.sum())
    n_pred = int(y_pred_bin.sum())
    print(f"  [{split:5s}] {target}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  MAE={mae:.4f}  RMSE={rmse:.4f}")
    print(f"         실제={n_true}  예측={n_pred}")

    return {
        "split": split, "target": target,
        "precision": prec, "recall": rec, "f1": f1,
        "mae": mae, "rmse": rmse,
        "n_true": n_true, "n_pred": n_pred,
    }


# Hurdle 파이프라인 — 분류기(발생 여부) → 회귀기(발생 크기) → 결합 평가 → SHAP
def run_hurdle(
    X_train, y_train,
    X_valid, y_valid,
    X_test,  y_test,
    target: str,
    base_params: dict,
    do_tune: bool,
    n_trials: int,
    dirs: dict,
    threshold: float,
) -> list[dict]:

    print(f"\n{'='*55}")
    print(f"  {target}  (hurdle, threshold={threshold})")
    print(f"{'='*55}")

    y_train_bin = (np.array(y_train) > threshold).astype(int)
    y_valid_bin = (np.array(y_valid) > threshold).astype(int)

    for label, yb in [("train", y_train_bin), ("valid", y_valid_bin)]:
        print(f"  {label} 발생 비율: {yb.mean():.3f} ({yb.sum()}/{len(yb)})")

    # Stage 1 — 분류기
    print(f"\n--- Stage 1: 분류기 ---")
    if do_tune:
        clf_params = tune_hurdle_clf(
            X_train, y_train_bin, X_valid, y_valid_bin,
            target, n_trials, base_params, dirs,
        )
    else:
        clf_params = _make_clf_params(base_params)

    clf = train_hurdle_clf(X_train, y_train_bin, X_valid, y_valid_bin, target, clf_params)

    # Stage 2 — 회귀기 (발생 케이스만)
    print(f"\n--- Stage 2: 회귀기 (발생 케이스만) ---")
    pos_train = y_train_bin == 1
    pos_valid = y_valid_bin == 1
    print(f"  train 발생: {pos_train.sum()}  valid 발생: {pos_valid.sum()}")

    reg_target = f"{target}_reg"
    if do_tune:
        reg_params = tune_params(
            X_train[pos_train], y_train[pos_train],
            X_valid[pos_valid], y_valid[pos_valid],
            reg_target, n_trials, base_params, dirs,
        )
    else:
        reg_params = base_params

    reg = train_lgbm(
        X_train[pos_train], y_train[pos_train],
        X_valid[pos_valid], y_valid[pos_valid],
        reg_target, reg_params,
    )

    # 결합 예측 및 평가
    print(f"\n--- 결합 평가 ---")
    results = []
    for split, X, y in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test",  X_test,  y_test),
    ]:
        pred_bin = clf.predict(X)
        pred_reg = reg.predict(X)
        pred_combined = np.where(pred_bin == 1, pred_reg, 0.0)
        results.append(evaluate_hurdle(y, pred_bin, pred_combined, split, target, threshold))

    shap_analysis(clf, X_test, f"{target}_clf", dirs)
    shap_analysis(reg, X_test, f"{target}_reg", dirs)

    clf.booster_.save_model(os.path.join(dirs["models"], f"lgbm_{target}_clf.txt"))
    reg.booster_.save_model(os.path.join(dirs["models"], f"lgbm_{target}_reg.txt"))

    return results


# 메인 — CSV 로드 → 분할 → 타겟별 학습 실행
def main(csv_path: str, target: str, do_tune: bool, n_trials: int, train_stride: int = 3, mask_names: list = None, hurdle: float = None, combo: str = None):
    # combo 우선 — MASK_COMBOS에서 사전 정의 조합 호출, mask_names 덮어쓰기
    combo_label = "baseline"
    if combo is not None:
        if combo not in MASK_COMBOS:
            raise ValueError(f"unknown combo: {combo}. 사용 가능: {list(MASK_COMBOS.keys())[:5]}...")
        mask_names = MASK_COMBOS[combo] or None
        combo_label = combo
    elif mask_names:
        combo_label = "mask_" + "_".join(sorted(mask_names))

    dirs = _make_run_dir(target, combo_label)
    print(f"output → {os.path.dirname(dirs['figures'])}")
    print(f"target: {target}  combo: {combo_label}")

    df = load_csv(csv_path)
    print(f"shape: {df.shape}")

    train_df, valid_df, test_df = split_data(df, train_stride=train_stride)

    # 마스크 적용 — 여러 마스크 조합 가능
    if mask_names:
        mask_set = set()
        for name in mask_names:
            mask_set.update(MASK_REGISTRY[name])
        feat_cols = [c for c in FEAT_COLS if c not in mask_set]
        print(f"features: {len(feat_cols)} (masked {len(FEAT_COLS) - len(feat_cols)}, groups: {mask_names})")
    else:
        feat_cols = FEAT_COLS
        print(f"features: {len(feat_cols)}")
    print(f"train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")

    X_train = train_df[feat_cols]
    X_valid = valid_df[feat_cols]
    X_test  = test_df[feat_cols]

    base_params = {**DEFAULT_PARAMS}

    # hurdle 모드 — 원래 스케일에서 threshold 비교, log 변환 스킵
    if hurdle is not None:
        y_train = train_df[target]
        y_valid = valid_df[target]
        y_test  = test_df[target]
        print(f"hurdle 모드 (threshold={hurdle})")

        results = run_hurdle(
            X_train, y_train,
            X_valid, y_valid,
            X_test,  y_test,
            target      = target,
            base_params = base_params,
            do_tune     = do_tune,
            n_trials    = n_trials,
            dirs        = dirs,
            threshold   = hurdle,
        )
    else:
        # 기존 회귀 모드
        cfg = TARGET_CONFIG.get(target, {"log": False})
        use_log = cfg.get("log", False)

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
        help="학습 타겟 (buzz_composite/growth/sustainability/crash + _5d/_10d/_15d, 또는 all)",
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
    parser.add_argument(
        "--mask", nargs="*", default=None, metavar="NAME",
        help="피처 마스크 적용 (momentum, buzz, noise, dead, wave, surge 조합 가능)",
    )
    parser.add_argument(
        "--combo", type=str, default=None,
        help="MASK_COMBOS 사전 정의 조합 이름 (예: S_buzz, D_buzz_noise, baseline)",
    )
    parser.add_argument(
        "--hurdle", type=float, default=None,
        help="Hurdle 모델 활성화, 값은 발생 판정 threshold (예: --hurdle 0.1)",
    )
    args = parser.parse_args()
    if args.gpu:
        DEFAULT_PARAMS["device"] = "cuda"
        print("GPU mode enabled")

    ALL_TARGETS = [
        "buzz_composite_5d", "buzz_composite_10d", "buzz_composite_15d",
        "growth_5d", "growth_10d", "growth_15d",
        "sustainability_5d", "sustainability_10d", "sustainability_15d",
        "crash_5d", "crash_10d", "crash_15d",
        "spike_5d", "spike_10d", "spike_15d",
    ]
    targets = ALL_TARGETS if args.target == "all" else [args.target]
    for t in targets:
        main(args.data, t, args.tune, args.n_trials, args.train_stride, args.mask, args.hurdle, args.combo)

