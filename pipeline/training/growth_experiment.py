# growth surge 구간 개선 실험 — threshold / min_child_samples / num_leaves 비교
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
    "baseline":               {"thresholds": [0.90, 1.56], "surge_min_child": 200, "surge_num_leaves": 64},
    "t1.4+child_50":          {"thresholds": [0.90, 1.40], "surge_min_child":  50, "surge_num_leaves": 64},
    "t1.3+child_50":          {"thresholds": [0.90, 1.30], "surge_min_child":  50, "surge_num_leaves": 64},
    "t1.4+child_50+leaves_32":{"thresholds": [0.90, 1.40], "surge_min_child":  50, "surge_num_leaves": 32},
    "t1.3+child_50+leaves_32":{"thresholds": [0.90, 1.30], "surge_min_child":  50, "surge_num_leaves": 32},
    "t1.4+child_30+leaves_32":{"thresholds": [0.90, 1.40], "surge_min_child":  30, "surge_num_leaves": 32},
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
    thresholds       = cfg["thresholds"]
    surge_min_child  = cfg["surge_min_child"]
    surge_num_leaves = cfg["surge_num_leaves"]
    names = ["shrink", "stable", "surge"]

    b_tr = make_bucket(y_tr, thresholds)
    b_va = make_bucket(y_va, thresholds)
    b_te = make_bucket(y_te, thresholds)
    w_tr = np.where((b_tr == 0) | (b_tr == 2), 2.0, 1.0)

    clf = lgb.LGBMClassifier(**CLF_BASE)
    clf.fit(X_tr, b_tr, sample_weight=w_tr, eval_set=[(X_va, b_va)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])

    regressors = {}
    fallbacks  = {}
    for b_idx, name in enumerate(names):
        mask_tr = b_tr == b_idx
        mask_va = b_va == b_idx
        fallbacks[b_idx] = float(np.median(y_tr[mask_tr])) if mask_tr.sum() > 0 else 0.0
        if mask_tr.sum() < 10:
            regressors[b_idx] = None
            continue
        reg_params = {**REG_BASE}
        if name == "surge":
            reg_params["min_child_samples"] = surge_min_child
            reg_params["num_leaves"]        = surge_num_leaves
        y_tr_b = np.log1p(y_tr[mask_tr])
        xv_b   = X_va[mask_va] if mask_va.sum() >= 5 else X_tr[mask_tr].iloc[:max(1, mask_tr.sum()//5)]
        yv_b   = np.log1p(y_va[mask_va]) if mask_va.sum() >= 5 else y_tr_b[:max(1, mask_tr.sum()//5)]
        reg = lgb.LGBMRegressor(**reg_params)
        reg.fit(X_tr[mask_tr], y_tr_b, eval_set=[(xv_b, yv_b)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        regressors[b_idx] = reg

    def predict(X):
        proba = clf.predict_proba(X)
        preds = np.zeros(len(X))
        for b_idx, model in regressors.items():
            rp = np.full(len(X), fallbacks[b_idx]) if model is None else np.expm1(model.predict(X))
            preds += proba[:, b_idx] * rp
        return preds

    print(f"\n[{exp_name}]  thresholds={thresholds}  surge_min_child={surge_min_child}  surge_num_leaves={surge_num_leaves}")
    rows = []
    for split_name, X, y, b in [("train", X_tr, y_tr, b_tr), ("valid", X_va, y_va, b_va), ("test", X_te, y_te, b_te)]:
        pred     = predict(X)
        mae_all  = float(mean_absolute_error(y, pred))
        rmse_all = rmse(y, pred)
        mae_per  = {}
        print(f"  {split_name:5s}  MAE={mae_all:.4f}  RMSE={rmse_all:.4f}")
        for b_idx, name in enumerate(names):
            mask  = b == b_idx
            mae_b = float(mean_absolute_error(y[mask], pred[mask])) if mask.sum() > 0 else 0.0
            mae_per[name] = mae_b
            print(f"    [{name}] MAE={mae_b:.4f}  n={mask.sum()}")
        rows.append({
            "exp": exp_name, "split": split_name,
            "mae": mae_all, "rmse": rmse_all,
            "mae_shrink": mae_per.get("shrink", 0),
            "mae_stable": mae_per.get("stable", 0),
            "mae_surge":  mae_per.get("surge",  0),
        })
    return rows


def main():
    df = load_csv(str(ROOT / "data_processed" / "dataset.csv"))
    train_df, valid_df, test_df = split_data(df)

    X_tr = train_df[FEAT_COLS]
    X_va = valid_df[FEAT_COLS]
    X_te = test_df[FEAT_COLS]

    target = "growth_5d"
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
    print(f"{'실험':<30} {'MAE':>8} {'RMSE':>8} {'shrink':>10} {'stable':>10} {'surge':>10}")
    print("-" * 82)
    for _, row in test_res.iterrows():
        print(f"{row['exp']:<30} {row['mae']:>8.4f} {row['rmse']:>8.4f} {row['mae_shrink']:>10.4f} {row['mae_stable']:>10.4f} {row['mae_surge']:>10.4f}")


if __name__ == "__main__":
    main()
