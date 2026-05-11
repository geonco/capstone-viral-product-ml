# buzz_composite single vs staged3 비교 실험
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pipeline.config import ROOT, FEAT_COLS, DATE_COL, TRAIN_RATIO, VALID_RATIO, GAP_DAYS
from pipeline.training.data_loader import load_csv

warnings.filterwarnings("ignore")

THRESHOLDS = [-0.46, 1.02]
NAMES      = ["negative", "neutral", "positive"]

CLF_PARAMS = {
    "boosting_type": "gbdt", "objective": "multiclass", "num_class": 3,
    "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 64, "max_depth": -1,
    "min_child_samples": 200, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42, "n_jobs": -1, "verbose": -1,
    "metric": "multi_logloss",
}

REG_PARAMS = {
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
    valid_end   = remaining[int(len(remaining) * VALID_RATIO / (1 - TRAIN_RATIO))]
    test_start  = valid_end + pd.Timedelta(days=GAP_DAYS)
    train = df[df[DATE_COL] <= train_end]
    valid = df[(df[DATE_COL] >= valid_start) & (df[DATE_COL] <= valid_end)]
    test  = df[df[DATE_COL] >= test_start]
    tr_dates = np.sort(train[DATE_COL].unique())
    va_dates = np.sort(valid[DATE_COL].unique())
    train = train[train[DATE_COL].isin(tr_dates[::3])]
    valid = valid[valid[DATE_COL].isin(va_dates[::3])]
    return train, valid, test


def make_bucket(y):
    t1, t2 = THRESHOLDS
    return np.where(y <= t1, 0, np.where(y <= t2, 1, 2))


def run_single(X_tr, y_tr, X_va, y_va, X_te, y_te):
    # 단일 회귀 — buzz_composite 현재 방식
    model = lgb.LGBMRegressor(**REG_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])

    rows = []
    for split_name, X, y in [("train", X_tr, y_tr), ("valid", X_va, y_va), ("test", X_te, y_te)]:
        pred = model.predict(X)
        b    = make_bucket(y)
        row  = {"mode": "single", "split": split_name,
                "mae": float(mean_absolute_error(y, pred)), "rmse": rmse(y, pred)}
        for b_idx, name in enumerate(NAMES):
            mask = b == b_idx
            row[f"mae_{name}"] = float(mean_absolute_error(y[mask], pred[mask])) if mask.sum() > 0 else None
        rows.append(row)
    return rows


def run_staged(X_tr, y_tr, X_va, y_va, X_te, y_te):
    # staged3 — 분류기 + 구간별 회귀기
    b_tr = make_bucket(y_tr)
    b_va = make_bucket(y_va)
    b_te = make_bucket(y_te)
    w_tr = np.where((b_tr == 0) | (b_tr == 2), 2.0, 1.0)

    clf = lgb.LGBMClassifier(**CLF_PARAMS)
    clf.fit(X_tr, b_tr, sample_weight=w_tr, eval_set=[(X_va, b_va)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    print(f"    clf valid accuracy: {float(np.mean(clf.predict(X_va) == b_va)):.4f}")

    regressors = {}
    fallbacks  = {}
    for b_idx, name in enumerate(NAMES):
        mask_tr = b_tr == b_idx
        mask_va = b_va == b_idx
        fallbacks[b_idx] = float(np.median(y_tr[mask_tr])) if mask_tr.sum() > 0 else 0.0
        print(f"    [{name}] tr={mask_tr.sum()}  va={mask_va.sum()}")
        if mask_tr.sum() < 10:
            regressors[b_idx] = None
            continue
        xv_b = X_va[mask_va] if mask_va.sum() >= 5 else X_tr[mask_tr].iloc[:max(1, mask_tr.sum()//5)]
        yv_b = y_va[mask_va] if mask_va.sum() >= 5 else y_tr[mask_tr][:max(1, mask_tr.sum()//5)]
        reg = lgb.LGBMRegressor(**REG_PARAMS)
        reg.fit(X_tr[mask_tr], y_tr[mask_tr], eval_set=[(xv_b, yv_b)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        regressors[b_idx] = reg

    rows = []
    for split_name, X, y, b in [("train", X_tr, y_tr, b_tr),
                                  ("valid", X_va, y_va, b_va),
                                  ("test",  X_te, y_te, b_te)]:
        proba = clf.predict_proba(X)
        pred  = np.zeros(len(X))
        for b_idx, model in regressors.items():
            rp = np.full(len(X), fallbacks[b_idx]) if model is None else model.predict(X)
            pred += proba[:, b_idx] * rp

        row = {"mode": "staged3", "split": split_name,
               "mae": float(mean_absolute_error(y, pred)), "rmse": rmse(y, pred)}
        for b_idx, name in enumerate(NAMES):
            mask = b == b_idx
            row[f"mae_{name}"] = float(mean_absolute_error(y[mask], pred[mask])) if mask.sum() > 0 else None
        rows.append(row)
    return rows


def print_table(title, df_split):
    print(f"\n{'='*85}")
    print(f"=== {title} ===")
    print(f"{'모드':<10} {'MAE':>8} {'RMSE':>8} {'negative':>10} {'neutral':>10} {'positive':>10}")
    print("-"*85)
    for _, row in df_split.iterrows():
        print(f"{row['mode']:<10} {row['mae']:>8.4f} {row['rmse']:>8.4f} "
              f"{row['mae_negative']:>10.4f} {row['mae_neutral']:>10.4f} {row['mae_positive']:>10.4f}")


def main():
    df = load_csv(str(ROOT / "data_processed" / "dataset.csv"))
    train_df, valid_df, test_df = split_data(df)

    X_tr = train_df[FEAT_COLS]
    X_va = valid_df[FEAT_COLS]
    X_te = test_df[FEAT_COLS]

    all_rows = []
    for target in ["buzz_composite_5d", "buzz_composite_10d", "buzz_composite_15d"]:
        print(f"\n{'#'*60}")
        print(f"# {target}")
        print(f"{'#'*60}")
        y_tr = train_df[target].values
        y_va = valid_df[target].values
        y_te = test_df[target].values

        print("\n[single]")
        rows = run_single(X_tr, y_tr, X_va, y_va, X_te, y_te)
        for r in rows:
            r["target"] = target
        all_rows.extend(rows)

        print("\n[staged3]")
        rows = run_staged(X_tr, y_tr, X_va, y_va, X_te, y_te)
        for r in rows:
            r["target"] = target
        all_rows.extend(rows)

    res = pd.DataFrame(all_rows)
    out_path = str(ROOT / "outputs" / "buzz_compare.csv")
    res.to_csv(out_path, index=False)
    print(f"\n결과 저장: {out_path}")

    for target in ["buzz_composite_5d", "buzz_composite_10d", "buzz_composite_15d"]:
        sub = res[res["target"] == target]
        print_table(f"{target} — test",  sub[sub["split"] == "test"])
        print_table(f"{target} — valid", sub[sub["split"] == "valid"])


if __name__ == "__main__":
    main()
