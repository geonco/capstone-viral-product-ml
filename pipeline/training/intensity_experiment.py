# intensity high 구간 개선 실험 — threshold / min_child_samples / EXTREME_WEIGHT 비교
import os
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import ROOT, FEAT_COLS, DATE_COL, TRAIN_RATIO, GAP_DAYS
from pipeline.training.data_loader import load_csv

warnings.filterwarnings("ignore")

EXPERIMENTS = {
    "best_baseline":              {"thresholds": [569.0,  800.0], "extreme_weight": 2.0, "high_min_child": 50,  "high_num_leaves": 64,  "high_n_estimators": 1000},
    "t750+child_50":              {"thresholds": [569.0,  750.0], "extreme_weight": 2.0, "high_min_child": 50,  "high_num_leaves": 64,  "high_n_estimators": 1000},
    "t800+leaves_32":             {"thresholds": [569.0,  800.0], "extreme_weight": 2.0, "high_min_child": 50,  "high_num_leaves": 32,  "high_n_estimators": 1000},
    "t800+leaves_16":             {"thresholds": [569.0,  800.0], "extreme_weight": 2.0, "high_min_child": 50,  "high_num_leaves": 16,  "high_n_estimators": 1000},
    "t800+estimators_2000":       {"thresholds": [569.0,  800.0], "extreme_weight": 2.0, "high_min_child": 50,  "high_num_leaves": 64,  "high_n_estimators": 2000},
    "t800+leaves_32+est_2000":    {"thresholds": [569.0,  800.0], "extreme_weight": 2.0, "high_min_child": 50,  "high_num_leaves": 32,  "high_n_estimators": 2000},
}

CLF_BASE = {
    "boosting_type": "gbdt", "objective": "multiclass", "num_class": 3,
    "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 64, "max_depth": -1,
    "min_child_samples": 200, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42, "n_jobs": -1, "verbose": -1,
    "metric": "multi_logloss",
}
REG_BASE = {
    "boosting_type": "gbdt", "objective": "regression",
    "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 64, "max_depth": -1,
    "min_child_samples": 200, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42, "n_jobs": -1, "verbose": -1,
}


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def split_data(df):
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    dates = np.sort(df[DATE_COL].unique())
    n = len(dates)
    train_end   = dates[int(n * TRAIN_RATIO)]
    valid_start = train_end + pd.Timedelta(days=GAP_DAYS)
    remaining   = dates[dates >= valid_start]
    valid_end   = remaining[int(len(remaining) * 0.5)]
    test_start  = valid_end + pd.Timedelta(days=GAP_DAYS)
    train = df[df[DATE_COL] <= train_end]
    valid = df[(df[DATE_COL] >= valid_start) & (df[DATE_COL] <= valid_end)]
    test  = df[df[DATE_COL] >= test_start]
    tr_dates = np.sort(train[DATE_COL].unique())
    va_dates = np.sort(valid[DATE_COL].unique())
    train = train[train[DATE_COL].isin(tr_dates[::3])]
    valid = valid[valid[DATE_COL].isin(va_dates[::3])]
    return train, valid, test


def make_bucket(y, thresholds):
    t1, t2 = thresholds
    return np.where(y <= t1, 0, np.where(y <= t2, 1, 2))


def run_experiment(exp_name, cfg, X_tr, y_tr, X_va, y_va, X_te, y_te):
    thresholds          = cfg["thresholds"]
    extreme_w           = cfg["extreme_weight"]
    high_min_child      = cfg["high_min_child"]
    high_num_leaves     = cfg.get("high_num_leaves", 64)
    high_n_estimators   = cfg.get("high_n_estimators", 1000)
    names = ["low", "mid", "high"]

    b_tr = make_bucket(y_tr, thresholds)
    b_va = make_bucket(y_va, thresholds)
    b_te = make_bucket(y_te, thresholds)

    # 샘플 가중치 — high(2), low(2), mid(1)
    w_tr = np.where((b_tr == 0) | (b_tr == 2), extreme_w, 1.0)

    # 분류기
    clf = lgb.LGBMClassifier(**CLF_BASE)
    clf.fit(X_tr, b_tr, sample_weight=w_tr, eval_set=[(X_va, b_va)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])

    # 구간별 회귀
    regressors = {}
    fallbacks  = {}
    for b_idx, name in enumerate(names):
        mask_tr = b_tr == b_idx
        mask_va = b_va == b_idx
        fallbacks[b_idx] = float(np.median(y_tr[mask_tr])) if mask_tr.sum() > 0 else 0.0
        if mask_tr.sum() < 10:
            regressors[b_idx] = None
            continue
        # high 구간은 min_child_samples / num_leaves / n_estimators 별도 적용
        reg_params = {**REG_BASE}
        if name == "high":
            reg_params["min_child_samples"] = high_min_child
            reg_params["num_leaves"]        = high_num_leaves
            reg_params["n_estimators"]      = high_n_estimators
        y_tr_b = np.log1p(y_tr[mask_tr])
        xv_b   = X_va[mask_va] if mask_va.sum() >= 5 else X_tr[mask_tr].iloc[:max(1, mask_tr.sum()//5)]
        yv_b   = np.log1p(y_va[mask_va]) if mask_va.sum() >= 5 else y_tr_b[:max(1, mask_tr.sum()//5)]
        reg = lgb.LGBMRegressor(**reg_params)
        reg.fit(X_tr[mask_tr], y_tr_b, eval_set=[(xv_b, yv_b)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        regressors[b_idx] = reg

    # soft predict
    def predict(X):
        proba = clf.predict_proba(X)
        preds = np.zeros(len(X))
        for b_idx, model in regressors.items():
            rp = np.full(len(X), fallbacks[b_idx]) if model is None else np.expm1(model.predict(X))
            preds += proba[:, b_idx] * rp
        return preds

    print(f"\n[{exp_name}]  thresholds={thresholds}  extreme_w={extreme_w}  high_min_child={high_min_child}")
    rows = []
    for split_name, X, y, b in [("train", X_tr, y_tr, b_tr), ("valid", X_va, y_va, b_va), ("test", X_te, y_te, b_te)]:
        pred = predict(X)
        mae_all  = float(mean_absolute_error(y, pred))
        rmse_all = rmse(y, pred)
        mae_per  = {}
        print(f"  {split_name:5s}  MAE={mae_all:.1f}  RMSE={rmse_all:.1f}")
        for b_idx, name in enumerate(names):
            mask = b == b_idx
            mae_b = float(mean_absolute_error(y[mask], pred[mask])) if mask.sum() > 0 else 0.0
            mae_per[name] = mae_b
            print(f"    [{name}] MAE={mae_b:.1f}  n={mask.sum()}")
        rows.append({
            "exp": exp_name, "split": split_name,
            "mae": mae_all, "rmse": rmse_all,
            "mae_low": mae_per.get("low", 0),
            "mae_mid": mae_per.get("mid", 0),
            "mae_high": mae_per.get("high", 0),
        })
    return rows


def main():
    df = load_csv(str(ROOT / "data_processed" / "dataset.csv"))
    train_df, valid_df, test_df = split_data(df)

    X_tr = train_df[FEAT_COLS]
    X_va = valid_df[FEAT_COLS]
    X_te = test_df[FEAT_COLS]

    target = "intensity_5d"
    y_tr = train_df[target].values
    y_va = valid_df[target].values
    y_te = test_df[target].values

    all_rows = []
    for exp_name, cfg in EXPERIMENTS.items():
        rows = run_experiment(exp_name, cfg, X_tr, y_tr, X_va, y_va, X_te, y_te)
        all_rows.extend(rows)

    print("\n\n===== 최종 비교 (test set) =====")
    res = pd.DataFrame(all_rows)
    test_res = res[res["split"] == "test"].copy()
    print(f"{'실험':<30} {'MAE':>8} {'RMSE':>8} {'low MAE':>10} {'mid MAE':>10} {'high MAE':>10}")
    print("-" * 80)
    for _, row in test_res.iterrows():
        print(f"{row['exp']:<30} {row['mae']:>8.1f} {row['rmse']:>8.1f} {row['mae_low']:>10.1f} {row['mae_mid']:>10.1f} {row['mae_high']:>10.1f}")


if __name__ == "__main__":
    main()
