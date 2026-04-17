# growth surge 극단값 개선 실험 — threshold 하향 + 4구간 + quantile 조합
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

# 실험 정의
# mode: "3class" | "4class"
# surge_obj: "regression" | "quantile"
# surge_alpha: quantile alpha (mode=quantile일 때)
# extreme_obj: "regression" | "quantile" (4class일 때)
EXPERIMENTS = {
    # 기준선 — 이전 q_alpha90 재현
    "baseline_q90":        {"mode": "3class", "t1": 0.90, "t2": 1.56,
                             "surge_obj": "quantile", "surge_alpha": 0.90},
    # threshold 낮춤 + q90
    "t1.3_q90":            {"mode": "3class", "t1": 0.90, "t2": 1.30,
                             "surge_obj": "quantile", "surge_alpha": 0.90},
    "t1.4_q90":            {"mode": "3class", "t1": 0.90, "t2": 1.40,
                             "surge_obj": "quantile", "surge_alpha": 0.90},
    # 4구간 — surge/extreme 분리, 둘 다 regression
    "4class_reg":          {"mode": "4class", "t1": 0.90, "t2": 1.40, "t3": 5.0,
                             "surge_obj": "regression",  "surge_alpha": None,
                             "extreme_obj": "regression", "extreme_alpha": None},
    # 4구간 — surge q90 + extreme regression
    "4class_sq90_ereg":    {"mode": "4class", "t1": 0.90, "t2": 1.40, "t3": 5.0,
                             "surge_obj": "quantile",   "surge_alpha": 0.90,
                             "extreme_obj": "regression", "extreme_alpha": None},
    # 4구간 — surge q90 + extreme q75
    "4class_sq90_eq75":    {"mode": "4class", "t1": 0.90, "t2": 1.40, "t3": 5.0,
                             "surge_obj": "quantile",   "surge_alpha": 0.90,
                             "extreme_obj": "quantile",  "extreme_alpha": 0.75},
    # 4구간 — surge q90 + extreme q90
    "4class_sq90_eq90":    {"mode": "4class", "t1": 0.90, "t2": 1.40, "t3": 5.0,
                             "surge_obj": "quantile",   "surge_alpha": 0.90,
                             "extreme_obj": "quantile",  "extreme_alpha": 0.90},
}

CLF_BASE = {
    "boosting_type": "gbdt", "objective": "multiclass",
    "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 64, "max_depth": -1,
    "min_child_samples": 200, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42, "n_jobs": -1, "verbose": -1,
    "metric": "multi_logloss",
}

REG_BASE = {
    "boosting_type": "gbdt",
    "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 32, "max_depth": -1,
    "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42, "n_jobs": -1, "verbose": -1,
}


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def pinball_loss(y, pred, alpha):
    err = y - pred
    return float(np.mean(np.where(err >= 0, alpha * err, (alpha - 1) * err)))


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


def make_bucket_3(y, t1, t2):
    return np.where(y <= t1, 0, np.where(y <= t2, 1, 2))


def make_bucket_4(y, t1, t2, t3):
    # 0:shrink / 1:stable / 2:surge / 3:extreme_surge
    return np.where(y <= t1, 0, np.where(y <= t2, 1, np.where(y <= t3, 2, 3)))


def fit_clf(X_tr, b_tr, X_va, b_va, n_class):
    # 분류기 — 양 끝 구간(소수 클래스)에 가중치 부여
    w_tr = np.where((b_tr == 0) | (b_tr == n_class - 1), 2.0, 1.0)
    params = {**CLF_BASE, "num_class": n_class}
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_tr, b_tr, sample_weight=w_tr, eval_set=[(X_va, b_va)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    return clf


def fit_reg(X_tr_b, y_tr_b, xv_b, yv_b, obj, alpha=None):
    # 구간별 회귀기 — objective와 alpha로 regression/quantile 분기
    params = {**REG_BASE, "objective": obj}
    if alpha is not None:
        params["alpha"] = alpha
    reg = lgb.LGBMRegressor(**params)
    reg.fit(X_tr_b, y_tr_b,
            eval_set=[(xv_b, yv_b)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    return reg


def soft_predict(clf, regressors, fallbacks, X):
    # clf 확률 가중합 — log1p 역변환 포함
    proba = clf.predict_proba(X)
    preds = np.zeros(len(X))
    for b_idx, model in regressors.items():
        rp = np.full(len(X), fallbacks[b_idx]) if model is None else np.expm1(model.predict(X))
        preds += proba[:, b_idx] * rp
    return preds


def _get_val_data(X_tr_b, y_tr_b, X_va, mask_va, y_va):
    # valid 샘플 부족 시 train 일부 대용
    if mask_va.sum() >= 5:
        return X_va[mask_va], np.log1p(y_va[mask_va])
    k = max(1, len(y_tr_b) // 5)
    return X_tr_b.iloc[:k], y_tr_b[:k]


def run_experiment(exp_name, cfg, X_tr, y_tr, X_va, y_va, X_te, y_te):
    mode = cfg["mode"]
    t1   = cfg["t1"]
    t2   = cfg["t2"]

    if mode == "3class":
        b_tr = make_bucket_3(y_tr, t1, t2)
        b_va = make_bucket_3(y_va, t1, t2)
        b_te = make_bucket_3(y_te, t1, t2)
        names   = ["shrink", "stable", "surge"]
        n_class = 3
    else:
        t3   = cfg["t3"]
        b_tr = make_bucket_4(y_tr, t1, t2, t3)
        b_va = make_bucket_4(y_va, t1, t2, t3)
        b_te = make_bucket_4(y_te, t1, t2, t3)
        names   = ["shrink", "stable", "surge", "extreme"]
        n_class = 4

    # 구간 분포 출력
    dist = "  ".join(f"{names[i]}={np.sum(b_te==i)}" for i in range(n_class))
    print(f"  test 분포: {dist}")

    clf = fit_clf(X_tr, b_tr, X_va, b_va, n_class)

    regressors = {}
    fallbacks  = {}

    for b_idx, name in enumerate(names):
        mask_tr = b_tr == b_idx
        mask_va = b_va == b_idx
        fallbacks[b_idx] = float(np.median(y_tr[mask_tr])) if mask_tr.sum() > 0 else 0.0

        print(f"    [{name}] train={mask_tr.sum()}  valid={mask_va.sum()}", end="")

        if mask_tr.sum() < 10:
            regressors[b_idx] = None
            print("  → fallback")
            continue

        y_tr_b = np.log1p(y_tr[mask_tr])
        xv_b, yv_b = _get_val_data(X_tr[mask_tr], y_tr_b, X_va, mask_va, y_va)

        if name == "surge":
            obj   = cfg["surge_obj"]
            alpha = cfg.get("surge_alpha")
        elif name == "extreme":
            obj   = cfg["extreme_obj"]
            alpha = cfg.get("extreme_alpha")
        else:
            obj, alpha = "regression", None

        reg = fit_reg(X_tr[mask_tr], y_tr_b, xv_b, yv_b, obj, alpha)
        regressors[b_idx] = reg
        print(f"  obj={obj}" + (f"  alpha={alpha}" if alpha else ""))

    # 평가
    rows = []
    for split_name, X, y, b in [("train", X_tr, y_tr, b_tr),
                                  ("valid", X_va, y_va, b_va),
                                  ("test",  X_te, y_te, b_te)]:
        pred = soft_predict(clf, regressors, fallbacks, X)

        mae_all  = float(mean_absolute_error(y, pred))
        rmse_all = rmse(y, pred)

        bucket_mae = {}
        for b_idx, name in enumerate(names):
            mask = b == b_idx
            bucket_mae[name] = float(mean_absolute_error(y[mask], pred[mask])) if mask.sum() > 0 else 0.0

        # surge + extreme 합산 (>=t2 전체)
        surge_ext_mask = b >= 2
        surge_ext_mae  = float(mean_absolute_error(y[surge_ext_mask], pred[surge_ext_mask])) if surge_ext_mask.sum() > 0 else 0.0
        pb90_surge_ext = pinball_loss(y[surge_ext_mask], pred[surge_ext_mask], 0.90) if surge_ext_mask.sum() > 0 else None

        rows.append({
            "exp":            exp_name,
            "mode":           mode,
            "split":          split_name,
            "mae":            mae_all,
            "rmse":           rmse_all,
            "mae_shrink":     bucket_mae.get("shrink",  0),
            "mae_stable":     bucket_mae.get("stable",  0),
            "mae_surge":      bucket_mae.get("surge",   0),
            "mae_extreme":    bucket_mae.get("extreme", None),
            "mae_surge_ext":  surge_ext_mae,
            "pb90_surge_ext": pb90_surge_ext,
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
        print(f"\n[{exp_name}]  mode={cfg['mode']}  t1={cfg['t1']}  t2={cfg['t2']}", end="")
        if cfg["mode"] == "4class":
            print(f"  t3={cfg['t3']}", end="")
        print(f"  surge_obj={cfg['surge_obj']}", end="")
        if cfg.get("surge_alpha"):
            print(f"(alpha={cfg['surge_alpha']})", end="")
        print()
        rows = run_experiment(exp_name, cfg, X_tr, y_tr, X_va, y_va, X_te, y_te)
        all_rows.extend(rows)

    res = pd.DataFrame(all_rows)
    out_path = str(ROOT / "outputs" / "growth_extreme_compare.csv")
    res.to_csv(out_path, index=False)

    test_res = res[res["split"] == "test"].copy()
    print(f"\n{'='*100}")
    print("=== test set 비교 ===")
    print(f"{'실험':<22} {'MAE':>8} {'RMSE':>8} {'shrink':>9} {'stable':>9} {'surge':>9} {'extreme':>9} {'surge+ext':>10} {'PB90':>8}")
    print("-"*100)
    for _, row in test_res.iterrows():
        ext   = f"{row['mae_extreme']:>9.4f}" if row['mae_extreme'] is not None else "        -"
        pb90  = f"{row['pb90_surge_ext']:>8.4f}" if row['pb90_surge_ext'] is not None else "       -"
        print(
            f"{row['exp']:<22} {row['mae']:>8.4f} {row['rmse']:>8.4f} "
            f"{row['mae_shrink']:>9.4f} {row['mae_stable']:>9.4f} "
            f"{row['mae_surge']:>9.4f} {ext} {row['mae_surge_ext']:>10.4f} {pb90}"
        )


if __name__ == "__main__":
    main()
