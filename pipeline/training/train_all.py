# 15개 타겟 전체 학습 파이프라인
# staged (분류+구간별 회귀): intensity, growth
# single (단일 회귀): crash, buzz_composite, sustainability

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import (
    ROOT, FEAT_COLS,
    DATE_COL, TRAIN_RATIO, VALID_RATIO, GAP_DAYS,
    TARGET_CONFIG, DEFAULT_PARAMS,
)
from pipeline.training.data_loader import load_csv

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# 타겟별 전략 — staged3: 3구간 / single: 단일 회귀
TARGET_STRATEGY = {
    "intensity_5d":       "staged3", "intensity_10d":       "staged3", "intensity_15d":       "staged3",
    "growth_5d":          "staged3", "growth_10d":          "staged3", "growth_15d":          "staged3",
    "crash_5d":           "single",  "crash_10d":           "single",  "crash_15d":           "single",
    "buzz_composite_5d":  "single",  "buzz_composite_10d":  "single",  "buzz_composite_15d":  "single",
    "sustainability_5d":  "single",  "sustainability_10d":  "single",  "sustainability_15d":  "single",
}

# 구간 설정 — 패밀리별 경계(원본 스케일), 구간명, 회귀 시 log1p 여부
BUCKET_CONFIG = {
    "intensity":     {"thresholds": [569.0, 800.0],       "names": ["low",    "mid",    "high"],                  "log": True,  "n_class": 3},
    "growth":        {"thresholds": [0.90,  1.56,  5.0],  "names": ["shrink", "stable", "surge", "extreme"],      "log": True,  "n_class": 4},
    "buzz_composite":{"thresholds": [-0.46, 1.02],        "names": ["negative", "neutral", "positive"],           "log": False, "n_class": 3},
}

# 구간별 회귀 파라미터 override — 패밀리+구간명 키로 DEFAULT_REG_PARAMS 일부를 덮어씀
# alphas 키가 있으면 해당 alpha 목록으로 quantile 앙상블 학습
BUCKET_REG_OVERRIDE = {
    ("intensity", "high"):   {"num_leaves": 32, "min_child_samples": 50},
    ("growth", "shrink"):    {"num_leaves": 32, "min_child_samples": 30},
    ("growth", "stable"):    {"num_leaves": 32, "min_child_samples": 30},
    ("growth", "surge"):     {"objective": "quantile", "num_leaves": 64, "min_child_samples": 20, "alphas": [0.75, 0.90]},
    ("growth", "extreme"):   {"objective": "quantile", "num_leaves": 16, "min_child_samples": 10, "alphas": [0.75, 0.90]},
}

# extreme 구간(양 끝) 샘플 가중치 — 소수 클래스 학습 보완
EXTREME_WEIGHT = 2.0

DEFAULT_CLF_PARAMS = {
    "boosting_type": "gbdt", "objective": "multiclass",
    "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 64, "max_depth": -1,
    "min_child_samples": 200, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42, "n_jobs": -1, "verbose": -1,
    "metric": "multi_logloss",
}

DEFAULT_REG_PARAMS = {
    "boosting_type": "gbdt", "objective": "regression",
    "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 64, "max_depth": -1,
    "min_child_samples": 200, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42, "n_jobs": -1, "verbose": -1,
}


def _make_run_dir():
    # 실험 결과 폴더 — figures/metrics/models 하위 디렉토리 생성
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run   = ROOT / "outputs" / f"run_{stamp}_all"
    dirs  = {k: str(run / k) for k in ["figures", "metrics", "models"]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs, str(run)


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _family(target):
    # intensity_5d → intensity, growth_10d → growth 등 패밀리명 추출
    return target.rsplit("_", 1)[0]


def make_bucket(y, thresholds):
    # 원본 스케일 경계 기준 구간 레이블 산출 — right=True로 경계값은 하위 구간에 포함
    return np.digitize(np.asarray(y), thresholds, right=True)


def make_weight(buckets, n_class):
    # 소수 클래스(양 끝 구간)에 가중치 부여
    return np.where((buckets == 0) | (buckets == n_class - 1), EXTREME_WEIGHT, 1.0)


def soft_predict(clf, regressors, fallbacks, X, use_log):
    # clf 확률 가중합으로 최종 예측 — 구간 경계 불연속 완화
    # regressors 값이 list이면 앙상블(평균), 단일 모델이면 그대로 사용
    proba = clf.predict_proba(X)
    preds = np.zeros(len(X))
    for b_idx, model in regressors.items():
        if model is None:
            reg_pred = np.full(len(X), fallbacks[b_idx])
        elif isinstance(model, list):
            raw = np.stack([m.predict(X) for m in model], axis=0).mean(axis=0)
            reg_pred = np.expm1(raw) if use_log else raw
        else:
            reg_pred = model.predict(X)
            if use_log:
                reg_pred = np.expm1(reg_pred)
        preds += proba[:, b_idx] * reg_pred
    return preds


def split_data(df, train_stride=1):
    # 시계열 기반 분할 — GAP_DAYS로 train/valid/test 간 누수 방지
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    dates = np.sort(df[DATE_COL].unique())
    n     = len(dates)
    train_end   = dates[int(n * TRAIN_RATIO)]
    valid_start = train_end + pd.Timedelta(days=GAP_DAYS)

    remaining = dates[dates >= valid_start]
    if len(remaining) == 0:
        raise ValueError("gap 적용 후 valid/test 데이터 없음")
    valid_end  = remaining[int(len(remaining) * VALID_RATIO / (1 - TRAIN_RATIO))]
    test_start = valid_end + pd.Timedelta(days=GAP_DAYS)

    train = df[df[DATE_COL] <= train_end]
    valid = df[(df[DATE_COL] >= valid_start) & (df[DATE_COL] <= valid_end)]
    test  = df[df[DATE_COL] >= test_start]

    if train_stride > 1:
        tr_dates = np.sort(train[DATE_COL].unique())
        va_dates = np.sort(valid[DATE_COL].unique())
        train = train[train[DATE_COL].isin(tr_dates[::train_stride])]
        valid = valid[valid[DATE_COL].isin(va_dates[::train_stride])]

    print(f"  train: {train[DATE_COL].min().date()} ~ {train[DATE_COL].max().date()} ({len(train):,})")
    print(f"  valid: {valid[DATE_COL].min().date()} ~ {valid[DATE_COL].max().date()} ({len(valid):,})")
    print(f"  test:  {test[DATE_COL].min().date()} ~ {test[DATE_COL].max().date()} ({len(test):,})")
    return train, valid, test


def _run_study(X_tr, y_tr, X_va, y_va, fixed, n_trials, w_tr, is_clf, label, dirs):
    # Optuna study — num_leaves/max_depth 탐색, trial별 best_iteration·과적합 기록
    trial_records = []

    def objective(trial):
        params = {
            **fixed,
            "n_estimators": 500,
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth":  trial.suggest_int("max_depth",   4,  10),
        }
        m = lgb.LGBMClassifier(**params) if is_clf else lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])

        best_iter   = m.best_iteration_ if hasattr(m, "best_iteration_") else -1
        pred_va     = m.predict(X_va)
        pred_tr     = m.predict(X_tr)
        score_va    = float(np.mean(pred_va != y_va)) if is_clf else rmse_score(y_va, pred_va)
        score_tr    = float(np.mean(pred_tr != y_tr)) if is_clf else rmse_score(y_tr, pred_tr)
        overfit_gap = round(score_va - score_tr, 6)  # 양수일수록 과적합

        trial_records.append({
            "trial":          trial.number,
            "num_leaves":     params["num_leaves"],
            "max_depth":      params["max_depth"],
            "learning_rate":  params["learning_rate"],
            "best_iteration": best_iter,
            "train_score":    round(score_tr, 6),
            "valid_score":    round(score_va, 6),
            "overfit_gap":    overfit_gap,
        })
        return score_va

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    (pd.DataFrame(trial_records)
     .sort_values("valid_score")
     .reset_index(drop=True)
     .to_csv(os.path.join(dirs["metrics"], f"tune_trials_{label}.csv"), index=False))

    return study.best_params, study.best_value


def _tune(X_tr, y_tr, X_va, y_va, fixed, n_trials, dirs, w_tr=None, label=""):
    # num_leaves / max_depth Optuna 탐색 — learning_rate는 DEFAULT_PARAMS에서 수동 설정
    is_clf = fixed.get("objective") in ("multiclass", "binary")
    lr     = fixed.get("learning_rate", 0.05)
    print(f"    [tune] {label}  lr={lr}  trials={n_trials}")

    best, val = _run_study(X_tr, y_tr, X_va, y_va, fixed, n_trials, w_tr, is_clf, label, dirs)
    print(f"      best: {val:.4f}  {best}")

    return {**fixed, **best}


def _shap_save(model, X, label, dirs):
    # SHAP 피처 중요도 — metrics 폴더에 CSV 저장
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # multiclass: (n_samples, n_features, n_class) 또는 list of 2D arrays
        if isinstance(sv, list):
            sv_abs = np.mean([np.abs(s) for s in sv], axis=0)
        elif sv.ndim == 3:
            sv_abs = np.abs(sv).mean(axis=2)
        else:
            sv_abs = np.abs(sv)
        (pd.DataFrame({"feature": X.columns, "mean_abs_shap": sv_abs.mean(0)})
         .sort_values("mean_abs_shap", ascending=False)
         .to_csv(os.path.join(dirs["metrics"], f"shap_{label}.csv"), index=False))
    except Exception as e:
        print(f"    SHAP 실패 ({label}): {e}")


def _evaluate(y_true, y_pred, split, target):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = rmse_score(y_true, y_pred)
    print(f"  [{split:5s}] MAE={mae:.4f}  RMSE={rmse:.4f}")
    return {"target": target, "split": split, "mae": mae, "rmse": rmse, "n": len(y_true)}


def run_staged(X_tr, y_tr, X_va, y_va, X_te, y_te, target, dirs, do_tune, n_trials):
    fam        = _family(target)
    cfg        = BUCKET_CONFIG[fam]
    thresholds = cfg["thresholds"]
    names      = cfg["names"]
    use_log    = cfg["log"]
    n_class    = cfg["n_class"]

    print(f"\n{'='*55}")
    print(f"  [staged] {target}  thresholds={thresholds}  log={use_log}  n_class={n_class}")
    print(f"{'='*55}")

    b_tr = make_bucket(y_tr, thresholds)
    b_va = make_bucket(y_va, thresholds)
    b_te = make_bucket(y_te, thresholds)
    w_tr = make_weight(b_tr, n_class)

    for split_name, b in [("train", b_tr), ("valid", b_va), ("test", b_te)]:
        dist = "  ".join(f"{names[i]}={np.sum(b==i)}({np.sum(b==i)/len(b)*100:.0f}%)" for i in range(n_class))
        print(f"  [{split_name}] {dist}")

    # STEP 1 — 분류기 학습
    print("\n  [STEP 1] Classifier")
    clf_params = {**DEFAULT_CLF_PARAMS, "num_class": n_class}
    if do_tune:
        clf_params = _tune(X_tr, b_tr, X_va, b_va, clf_params, n_trials, dirs, w_tr, label=f"{target}_clf")

    clf = lgb.LGBMClassifier(**clf_params)
    clf.fit(X_tr, b_tr, sample_weight=w_tr, eval_set=[(X_va, b_va)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    print(f"    valid accuracy: {float(np.mean(clf.predict(X_va) == b_va)):.4f}")
    clf.booster_.save_model(os.path.join(dirs["models"], f"lgbm_{target}_clf.txt"))
    _shap_save(clf, X_te.iloc[:min(3000, len(X_te))], f"{target}_clf", dirs)

    # STEP 2 — 구간별 회귀기 학습
    print("\n  [STEP 2] Per-bucket Regressors")
    regressors = {}
    fallbacks  = {}

    for b_idx, name in enumerate(names):
        mask_tr = b_tr == b_idx
        mask_va = b_va == b_idx
        fallbacks[b_idx] = float(np.median(y_tr[mask_tr])) if mask_tr.sum() > 0 else 0.0

        print(f"\n    [{name}] train={mask_tr.sum()}  valid={mask_va.sum()}")
        if mask_tr.sum() < 10:
            regressors[b_idx] = None
            continue

        y_tr_b = np.log1p(y_tr[mask_tr]) if use_log else y_tr[mask_tr]

        # valid 샘플 부족 시 train 일부를 대용
        if mask_va.sum() >= 5:
            xv_b = X_va[mask_va]
            yv_b = np.log1p(y_va[mask_va]) if use_log else y_va[mask_va]
        else:
            k    = max(1, mask_tr.sum() // 5)
            xv_b = X_tr[mask_tr].iloc[:k]
            yv_b = y_tr_b[:k]

        raw_override = BUCKET_REG_OVERRIDE.get((fam, name), {})
        alphas       = raw_override.get("alphas", None)
        override     = {k: v for k, v in raw_override.items() if k != "alphas"}
        reg_params  = {**DEFAULT_REG_PARAMS, **override}

        if alphas:
            # quantile 앙상블 — alpha별 모델 각각 학습 후 리스트로 저장
            regs = []
            for alpha in alphas:
                p = {**reg_params, "objective": "quantile", "alpha": alpha}
                r = lgb.LGBMRegressor(**p)
                r.fit(X_tr[mask_tr], y_tr_b, eval_set=[(xv_b, yv_b)],
                      callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
                r.booster_.save_model(os.path.join(dirs["models"], f"lgbm_{target}_reg_{name}_q{int(alpha*100)}.txt"))
                regs.append(r)
            regressors[b_idx] = regs
            te_mask = b_te == b_idx
            shap_X  = X_te[te_mask].iloc[:min(3000, te_mask.sum())] if te_mask.sum() > 0 else X_te.iloc[:1]
            _shap_save(regs[0], shap_X, f"{target}_reg_{name}", dirs)
        else:
            if do_tune:
                reg_params = _tune(X_tr[mask_tr], y_tr_b, xv_b, yv_b,
                                   reg_params, n_trials, dirs, label=f"{target}_reg_{name}")
            reg = lgb.LGBMRegressor(**reg_params)
            reg.fit(X_tr[mask_tr], y_tr_b, eval_set=[(xv_b, yv_b)],
                    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
            regressors[b_idx] = reg
            reg.booster_.save_model(os.path.join(dirs["models"], f"lgbm_{target}_reg_{name}.txt"))
            te_mask = b_te == b_idx
            shap_X  = X_te[te_mask].iloc[:min(3000, te_mask.sum())] if te_mask.sum() > 0 else X_te.iloc[:1]
            _shap_save(reg, shap_X, f"{target}_reg_{name}", dirs)

    # STEP 3 — soft routing 전체 평가 + 구간별 MAE
    print("\n  [STEP 3] 전체 평가")
    results = []
    for split_name, X, y, b in [("train", X_tr, y_tr, b_tr), ("valid", X_va, y_va, b_va), ("test", X_te, y_te, b_te)]:
        pred = soft_predict(clf, regressors, fallbacks, X, use_log)
        results.append(_evaluate(y, pred, split_name, target))
        # 구간별 MAE
        for b_idx, name in enumerate(names):
            mask = b == b_idx
            if mask.sum() > 0:
                mae_b = float(mean_absolute_error(y[mask], pred[mask]))
                print(f"    [{split_name}][{name}] MAE={mae_b:.4f}  n={mask.sum()}")

    return results


def run_single(X_tr, y_tr, X_va, y_va, X_te, y_te, target, dirs, do_tune, n_trials):
    use_log = TARGET_CONFIG.get(target, {}).get("log", False)

    print(f"\n{'='*55}")
    print(f"  [single] {target}  log={use_log}")
    print(f"{'='*55}")

    y_tr_t = np.log1p(y_tr) if use_log else y_tr
    y_va_t = np.log1p(y_va) if use_log else y_va

    params = {**DEFAULT_PARAMS}
    if do_tune:
        params = _tune(X_tr, y_tr_t, X_va, y_va_t, params, n_trials, dirs, label=target)

    model = lgb.LGBMRegressor(**params)
    model.fit(X_tr, y_tr_t, eval_set=[(X_va, y_va_t)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    print(f"  최적 트리 수: {model.best_iteration_}")
    model.booster_.save_model(os.path.join(dirs["models"], f"lgbm_{target}.txt"))
    _shap_save(model, X_te.iloc[:min(3000, len(X_te))], target, dirs)

    results = []
    for split_name, X, y_orig in [("train", X_tr, y_tr), ("valid", X_va, y_va), ("test", X_te, y_te)]:
        pred_t = model.predict(X)
        pred   = np.expm1(pred_t) if use_log else pred_t
        results.append(_evaluate(y_orig, pred, split_name, target))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",         type=str,
                        default=str(ROOT / "data_processed" / "dataset.csv"),
                        help="피처 + 라벨 CSV 경로")
    parser.add_argument("--targets",      type=str, default="all",
                        help="콤마 구분 타겟 목록 또는 all")
    parser.add_argument("--tune",         action="store_true",
                        help="Optuna TPE 하이퍼파라미터 튜닝 활성화")
    parser.add_argument("--n_trials",     type=int, default=25,
                        help="Optuna trial 횟수")
    parser.add_argument("--train_stride", type=int, default=3,
                        help="train/valid 서브샘플링 간격 (test는 stride=1 고정)")
    args = parser.parse_args()

    dirs, run_dir = _make_run_dir()
    print(f"output → {run_dir}")

    df = load_csv(args.data)
    print(f"shape: {df.shape}")

    print(f"\n시계열 분할 (gap={GAP_DAYS}일, stride={args.train_stride})")
    train_df, valid_df, test_df = split_data(df, args.train_stride)
    print(f"features: {len(FEAT_COLS)}")

    X_train = train_df[FEAT_COLS]
    X_valid = valid_df[FEAT_COLS]
    X_test  = test_df[FEAT_COLS]

    all_targets = list(TARGET_STRATEGY.keys())
    targets = all_targets if args.targets == "all" else [t.strip() for t in args.targets.split(",")]

    all_results = []
    for target in targets:
        strategy = TARGET_STRATEGY.get(target, "single")
        y_tr = train_df[target].values
        y_va = valid_df[target].values
        y_te = test_df[target].values

        if strategy == "staged3":
            results = run_staged(X_train, y_tr, X_valid, y_va, X_test, y_te,
                                 target, dirs, args.tune, args.n_trials)
        else:
            results = run_single(X_train, y_tr, X_valid, y_va, X_test, y_te,
                                 target, dirs, args.tune, args.n_trials)
        all_results.extend(results)

    res_df = pd.DataFrame(all_results)
    res_df.to_csv(os.path.join(dirs["metrics"], "evaluation_results.csv"), index=False)

    print(f"\n{'='*55}")
    print("전체 결과 (train / valid / test)")
    print("="*55)
    for target in res_df["target"].unique():
        sub = res_df[res_df["target"] == target]
        print(f"\n  {target}")
        for _, row in sub.iterrows():
            print(f"    {row['split']:5s}  MAE={row['mae']:.4f}  RMSE={row['rmse']:.4f}")


if __name__ == "__main__":
    main()
