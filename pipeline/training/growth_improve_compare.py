# growth surge 개선 실험 — 구간별 파라미터 튜닝 vs Quantile 앙상블 비교
# 기준선: 4class_sq90_eq90 (surge q90 + extreme q90)
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

THRESHOLDS = [0.90, 1.56, 5.0]   # shrink / stable / surge / extreme
NAMES      = ["shrink", "stable", "surge", "extreme"]

CLF_PARAMS = {
    "boosting_type": "gbdt", "objective": "multiclass", "num_class": 4,
    "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 64, "max_depth": -1,
    "min_child_samples": 200, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42, "n_jobs": -1, "verbose": -1,
    "metric": "multi_logloss",
}

# 구간별 회귀기 파라미터 — 실험마다 surge/extreme override
BASE_REG = {
    "boosting_type": "gbdt",
    "n_estimators": 1000, "learning_rate": 0.05,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42, "n_jobs": -1, "verbose": -1,
}

# shrink / stable 파라미터 고정
STABLE_REG = {**BASE_REG, "objective": "regression", "num_leaves": 32, "min_child_samples": 30}

# 실험 정의
# surge_params / extreme_params: 각 구간 회귀기 파라미터 (objective 포함)
# surge_alphas / extreme_alphas: list이면 앙상블, None이면 단일
EXPERIMENTS = {
    # 기준선
    "baseline":            {
        "surge_params":   {**BASE_REG, "objective": "quantile", "alpha": 0.90, "num_leaves": 32, "min_child_samples": 30},
        "extreme_params":  {**BASE_REG, "objective": "quantile", "alpha": 0.90, "num_leaves": 32, "min_child_samples": 30},
        "surge_alphas":    None, "extreme_alphas": None,
    },
    # 접근법 1 — 구간별 파라미터 튜닝
    # surge: 샘플 충분 → num_leaves 높여 복잡도 증가
    # extreme: 샘플 부족 → num_leaves 낮추고 min_child 낮춰 과적합 방지
    "pt_s64_e16":          {
        "surge_params":   {**BASE_REG, "objective": "quantile", "alpha": 0.90, "num_leaves": 64, "min_child_samples": 20},
        "extreme_params":  {**BASE_REG, "objective": "quantile", "alpha": 0.90, "num_leaves": 16, "min_child_samples": 10},
        "surge_alphas":    None, "extreme_alphas": None,
    },
    "pt_s128_e16":         {
        "surge_params":   {**BASE_REG, "objective": "quantile", "alpha": 0.90, "num_leaves": 128, "min_child_samples": 10},
        "extreme_params":  {**BASE_REG, "objective": "quantile", "alpha": 0.90, "num_leaves": 16, "min_child_samples": 10},
        "surge_alphas":    None, "extreme_alphas": None,
    },
    "pt_s128_e8":          {
        "surge_params":   {**BASE_REG, "objective": "quantile", "alpha": 0.90, "num_leaves": 128, "min_child_samples": 10},
        "extreme_params":  {**BASE_REG, "objective": "quantile", "alpha": 0.90, "num_leaves": 8,  "min_child_samples": 5},
        "surge_alphas":    None, "extreme_alphas": None,
    },
    # 접근법 2 — Quantile 앙상블 (surge+extreme 모두)
    # 여러 alpha 예측 평균 → 단일 alpha보다 안정적인 극단값 예측
    "qens_s_075_090":      {
        "surge_params":   {**BASE_REG, "num_leaves": 64, "min_child_samples": 20},
        "extreme_params":  {**BASE_REG, "num_leaves": 16, "min_child_samples": 10},
        "surge_alphas":    [0.75, 0.90], "extreme_alphas": [0.75, 0.90],
    },
    "qens_s_075_090_095":  {
        "surge_params":   {**BASE_REG, "num_leaves": 64, "min_child_samples": 20},
        "extreme_params":  {**BASE_REG, "num_leaves": 16, "min_child_samples": 10},
        "surge_alphas":    [0.75, 0.90, 0.95], "extreme_alphas": [0.75, 0.90, 0.95],
    },
    "qens_s_090_095_099":  {
        "surge_params":   {**BASE_REG, "num_leaves": 64, "min_child_samples": 20},
        "extreme_params":  {**BASE_REG, "num_leaves": 16, "min_child_samples": 10},
        "surge_alphas":    [0.90, 0.95, 0.99], "extreme_alphas": [0.90, 0.95, 0.99],
    },
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


def make_bucket(y):
    t1, t2, t3 = THRESHOLDS
    return np.where(y <= t1, 0, np.where(y <= t2, 1, np.where(y <= t3, 2, 3)))


def fit_clf(X_tr, b_tr, X_va, b_va):
    w_tr = np.where((b_tr == 0) | (b_tr == 3), 2.0, 1.0)
    clf = lgb.LGBMClassifier(**CLF_PARAMS)
    clf.fit(X_tr, b_tr, sample_weight=w_tr, eval_set=[(X_va, b_va)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    return clf


def fit_reg_single(X_tr_b, y_tr_b, xv_b, yv_b, params):
    reg = lgb.LGBMRegressor(**params)
    reg.fit(X_tr_b, y_tr_b, eval_set=[(xv_b, yv_b)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    return reg


def fit_reg_ensemble(X_tr_b, y_tr_b, xv_b, yv_b, base_params, alphas):
    # 여러 alpha로 각각 학습 후 리스트 반환 — 예측 시 평균
    models = []
    for alpha in alphas:
        p = {**base_params, "objective": "quantile", "alpha": alpha}
        reg = lgb.LGBMRegressor(**p)
        reg.fit(X_tr_b, y_tr_b, eval_set=[(xv_b, yv_b)],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        models.append(reg)
    return models


def get_val_data(X_tr_b, y_tr_b, X_va, mask_va, y_va):
    if mask_va.sum() >= 5:
        return X_va[mask_va], np.log1p(y_va[mask_va])
    k = max(1, len(y_tr_b) // 5)
    return X_tr_b.iloc[:k], y_tr_b[:k]


def predict_bucket(model_or_list, X):
    # 단일 모델 or 앙상블(list) 예측 — log1p 역변환 포함
    if isinstance(model_or_list, list):
        preds = np.stack([np.expm1(m.predict(X)) for m in model_or_list], axis=0)
        return preds.mean(axis=0)
    return np.expm1(model_or_list.predict(X))


def soft_predict(clf, regressors, fallbacks, X):
    proba = clf.predict_proba(X)
    preds = np.zeros(len(X))
    for b_idx in range(4):
        model = regressors[b_idx]
        rp = np.full(len(X), fallbacks[b_idx]) if model is None else predict_bucket(model, X)
        preds += proba[:, b_idx] * rp
    return preds


def run_experiment(exp_name, cfg, X_tr, y_tr, X_va, y_va, X_te, y_te):
    b_tr = make_bucket(y_tr)
    b_va = make_bucket(y_va)
    b_te = make_bucket(y_te)

    clf = fit_clf(X_tr, b_tr, X_va, b_va)

    regressors = {}
    fallbacks  = {}

    for b_idx, name in enumerate(NAMES):
        mask_tr = b_tr == b_idx
        mask_va = b_va == b_idx
        fallbacks[b_idx] = float(np.median(y_tr[mask_tr])) if mask_tr.sum() > 0 else 0.0

        print(f"    [{name}] tr={mask_tr.sum()} va={mask_va.sum()}", end="")

        if mask_tr.sum() < 10:
            regressors[b_idx] = None
            print(" → fallback")
            continue

        y_tr_b = np.log1p(y_tr[mask_tr])
        xv_b, yv_b = get_val_data(X_tr[mask_tr], y_tr_b, X_va, mask_va, y_va)

        if name in ("shrink", "stable"):
            reg = fit_reg_single(X_tr[mask_tr], y_tr_b, xv_b, yv_b, STABLE_REG)
            regressors[b_idx] = reg
            print(" → regression")

        elif name == "surge":
            alphas = cfg["surge_alphas"]
            if alphas:
                reg = fit_reg_ensemble(X_tr[mask_tr], y_tr_b, xv_b, yv_b, cfg["surge_params"], alphas)
                print(f" → ensemble {alphas}")
            else:
                reg = fit_reg_single(X_tr[mask_tr], y_tr_b, xv_b, yv_b, cfg["surge_params"])
                print(f" → q{cfg['surge_params'].get('alpha', 'reg')}")
            regressors[b_idx] = reg

        elif name == "extreme":
            alphas = cfg["extreme_alphas"]
            if alphas:
                reg = fit_reg_ensemble(X_tr[mask_tr], y_tr_b, xv_b, yv_b, cfg["extreme_params"], alphas)
                print(f" → ensemble {alphas}")
            else:
                reg = fit_reg_single(X_tr[mask_tr], y_tr_b, xv_b, yv_b, cfg["extreme_params"])
                print(f" → q{cfg['extreme_params'].get('alpha', 'reg')}")
            regressors[b_idx] = reg

    rows = []
    for split_name, X, y, b in [("train", X_tr, y_tr, b_tr),
                                  ("valid", X_va, y_va, b_va),
                                  ("test",  X_te, y_te, b_te)]:
        pred = soft_predict(clf, regressors, fallbacks, X)

        mae_all  = float(mean_absolute_error(y, pred))
        rmse_all = rmse(y, pred)

        bucket_mae = {}
        for b_idx, name in enumerate(NAMES):
            mask = b == b_idx
            bucket_mae[name] = float(mean_absolute_error(y[mask], pred[mask])) if mask.sum() > 0 else 0.0

        surge_ext_mask = b >= 2
        surge_ext_mae  = float(mean_absolute_error(y[surge_ext_mask], pred[surge_ext_mask])) if surge_ext_mask.sum() > 0 else 0.0
        pb90 = pinball_loss(y[surge_ext_mask], pred[surge_ext_mask], 0.90) if surge_ext_mask.sum() > 0 else None

        rows.append({
            "exp":           exp_name,
            "split":         split_name,
            "mae":           mae_all,
            "rmse":          rmse_all,
            "mae_shrink":    bucket_mae["shrink"],
            "mae_stable":    bucket_mae["stable"],
            "mae_surge":     bucket_mae["surge"],
            "mae_extreme":   bucket_mae["extreme"],
            "mae_surge_ext": surge_ext_mae,
            "pb90":          pb90,
        })

    return rows


def print_table(title, df_split):
    print(f"\n{'='*105}")
    print(f"=== {title} ===")
    print(f"{'실험':<24} {'MAE':>8} {'RMSE':>8} {'shrink':>9} {'stable':>9} {'surge':>9} {'extreme':>9} {'surge+ext':>10} {'PB90':>8}")
    print("-"*105)
    for _, row in df_split.iterrows():
        pb = f"{row['pb90']:>8.4f}" if row['pb90'] is not None else "       -"
        print(
            f"{row['exp']:<24} {row['mae']:>8.4f} {row['rmse']:>8.4f} "
            f"{row['mae_shrink']:>9.4f} {row['mae_stable']:>9.4f} "
            f"{row['mae_surge']:>9.4f} {row['mae_extreme']:>9.4f} "
            f"{row['mae_surge_ext']:>10.4f} {pb}"
        )


def main():
    df = load_csv(str(ROOT / "data_processed" / "dataset.csv"))
    train_df, valid_df, test_df = split_data(df)

    X_tr = train_df[FEAT_COLS]
    X_va = valid_df[FEAT_COLS]
    X_te = test_df[FEAT_COLS]

    all_rows = []
    for target in ["growth_5d", "growth_10d", "growth_15d"]:
        print(f"\n{'#'*60}")
        print(f"# TARGET: {target}")
        print(f"{'#'*60}")
        y_tr = train_df[target].values
        y_va = valid_df[target].values
        y_te = test_df[target].values

        for exp_name, cfg in EXPERIMENTS.items():
            print(f"\n[{target}][{exp_name}]")
            rows = run_experiment(exp_name, cfg, X_tr, y_tr, X_va, y_va, X_te, y_te)
            for r in rows:
                r["target"] = target
            all_rows.extend(rows)

    res = pd.DataFrame(all_rows)
    out_path = str(ROOT / "outputs" / "growth_improve_compare.csv")
    res.to_csv(out_path, index=False)
    print(f"\n결과 저장: {out_path}")

    for target in ["growth_5d", "growth_10d", "growth_15d"]:
        sub = res[res["target"] == target]
        print_table(f"{target} — test set",  sub[sub["split"] == "test"])
        print_table(f"{target} — valid set", sub[sub["split"] == "valid"])


if __name__ == "__main__":
    main()
