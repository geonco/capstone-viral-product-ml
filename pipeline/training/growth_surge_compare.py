# growth surge 구간 개선 실험 — Weighted 학습 vs Quantile Regression 비교
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

# 구간 경계
THRESHOLDS = [0.90, 1.56]
NAMES      = ["shrink", "stable", "surge"]

# 실험 정의 — mode: "weighted" | "quantile"
EXPERIMENTS = {
    "baseline":          {"mode": "baseline"},
    "w_surge5":          {"mode": "weighted",  "surge_weight": 5.0},
    "w_surge10":         {"mode": "weighted",  "surge_weight": 10.0},
    "w_surge20":         {"mode": "weighted",  "surge_weight": 20.0},
    "q_alpha50":         {"mode": "quantile",  "alpha": 0.50},
    "q_alpha75":         {"mode": "quantile",  "alpha": 0.75},
    "q_alpha90":         {"mode": "quantile",  "alpha": 0.90},
}

CLF_PARAMS = {
    "boosting_type": "gbdt", "objective": "multiclass", "num_class": 3,
    "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 64, "max_depth": -1,
    "min_child_samples": 200, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42, "n_jobs": -1, "verbose": -1,
    "metric": "multi_logloss",
}

REG_BASE = {
    "boosting_type": "gbdt",
    "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 64, "max_depth": -1,
    "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42, "n_jobs": -1, "verbose": -1,
}


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def pinball_loss(y, q_pred, alpha):
    # quantile 예측 오차 — 실제값과 예측값의 분위수 기반 손실
    err = y - q_pred
    return float(np.mean(np.where(err >= 0, alpha * err, (alpha - 1) * err)))


def weighted_mae(y, pred, b, surge_w=10.0):
    # surge 구간에 가중치를 부여한 MAE — surge 예측 오차를 강조해서 집계
    w = np.where(b == 2, surge_w, 1.0)
    return float(np.average(np.abs(y - pred), weights=w))


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
    t1, t2 = THRESHOLDS
    return np.where(y <= t1, 0, np.where(y <= t2, 1, 2))


def fit_clf(X_tr, b_tr, X_va, b_va):
    # 3구간 분류기 학습 — shrink/surge(양 끝)에 EXTREME_WEIGHT 부여
    w_tr = np.where((b_tr == 0) | (b_tr == 2), 2.0, 1.0)
    clf = lgb.LGBMClassifier(**CLF_PARAMS)
    clf.fit(X_tr, b_tr, sample_weight=w_tr, eval_set=[(X_va, b_va)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    return clf


def fit_reg_bucket(X_tr_b, y_tr_b, xv_b, yv_b, obj, alpha=None, sample_weight=None):
    # 구간별 회귀기 학습 — objective와 alpha로 baseline/weighted/quantile 분기
    params = {**REG_BASE, "objective": obj}
    if alpha is not None:
        params["alpha"] = alpha
    reg = lgb.LGBMRegressor(**params)
    reg.fit(X_tr_b, y_tr_b, sample_weight=sample_weight,
            eval_set=[(xv_b, yv_b)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    return reg


def soft_predict(clf, regressors, fallbacks, X):
    # clf 확률 가중합으로 최종 예측 — log1p 역변환 포함
    proba = clf.predict_proba(X)
    preds = np.zeros(len(X))
    for b_idx, model in regressors.items():
        if model is None:
            rp = np.full(len(X), fallbacks[b_idx])
        else:
            rp = np.expm1(model.predict(X))
        preds += proba[:, b_idx] * rp
    return preds


def run_experiment(exp_name, cfg, X_tr, y_tr, X_va, y_va, X_te, y_te):
    mode = cfg["mode"]

    b_tr = make_bucket(y_tr)
    b_va = make_bucket(y_va)
    b_te = make_bucket(y_te)

    # 분류기 — 모든 실험 공통
    clf = fit_clf(X_tr, b_tr, X_va, b_va)

    # 구간별 회귀기 학습
    regressors = {}
    fallbacks  = {}

    for b_idx, name in enumerate(NAMES):
        mask_tr = b_tr == b_idx
        mask_va = b_va == b_idx
        fallbacks[b_idx] = float(np.median(y_tr[mask_tr])) if mask_tr.sum() > 0 else 0.0

        if mask_tr.sum() < 10:
            regressors[b_idx] = None
            continue

        y_tr_b = np.log1p(y_tr[mask_tr])
        xv_b   = X_va[mask_va] if mask_va.sum() >= 5 else X_tr[mask_tr].iloc[:max(1, mask_tr.sum()//5)]
        yv_b   = np.log1p(y_va[mask_va]) if mask_va.sum() >= 5 else y_tr_b[:max(1, mask_tr.sum()//5)]

        if mode == "baseline":
            # 기본 regression, 가중치 없음
            reg = fit_reg_bucket(X_tr[mask_tr], y_tr_b, xv_b, yv_b, obj="regression")

        elif mode == "weighted":
            surge_w = cfg["surge_weight"]
            if name == "surge":
                # surge 구간 내 샘플 전체에 균일 가중치 적용
                sw = np.full(mask_tr.sum(), surge_w)
            else:
                sw = None
            reg = fit_reg_bucket(X_tr[mask_tr], y_tr_b, xv_b, yv_b,
                                  obj="regression", sample_weight=sw)

        elif mode == "quantile":
            alpha = cfg["alpha"]
            if name == "surge":
                # surge 구간만 quantile regression — 분위수 예측으로 극단값 대응
                reg = fit_reg_bucket(X_tr[mask_tr], y_tr_b, xv_b, yv_b,
                                      obj="quantile", alpha=alpha)
            else:
                reg = fit_reg_bucket(X_tr[mask_tr], y_tr_b, xv_b, yv_b,
                                      obj="regression")

        regressors[b_idx] = reg

    # 평가
    rows = []
    for split_name, X, y, b in [("train", X_tr, y_tr, b_tr),
                                  ("valid", X_va, y_va, b_va),
                                  ("test",  X_te, y_te, b_te)]:
        pred = soft_predict(clf, regressors, fallbacks, X)

        mae_all   = float(mean_absolute_error(y, pred))
        rmse_all  = rmse(y, pred)
        wt_mae    = weighted_mae(y, pred, b, surge_w=10.0)

        # quantile 실험이면 surge 구간에 대해 pinball loss 추가
        surge_mask = b == 2
        pb_50 = pb_75 = pb_90 = None
        if surge_mask.sum() > 0:
            pb_50 = pinball_loss(y[surge_mask], pred[surge_mask], 0.50)
            pb_75 = pinball_loss(y[surge_mask], pred[surge_mask], 0.75)
            pb_90 = pinball_loss(y[surge_mask], pred[surge_mask], 0.90)

        bucket_mae = {}
        for b_idx, name in enumerate(NAMES):
            mask = b == b_idx
            bucket_mae[name] = float(mean_absolute_error(y[mask], pred[mask])) if mask.sum() > 0 else 0.0

        rows.append({
            "exp":          exp_name,
            "mode":         mode,
            "split":        split_name,
            "mae":          mae_all,
            "rmse":         rmse_all,
            "weighted_mae": wt_mae,
            "mae_shrink":   bucket_mae["shrink"],
            "mae_stable":   bucket_mae["stable"],
            "mae_surge":    bucket_mae["surge"],
            "pinball_50":   pb_50,
            "pinball_75":   pb_75,
            "pinball_90":   pb_90,
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
        print(f"\n[{exp_name}] mode={cfg['mode']}", end="")
        if "surge_weight" in cfg:
            print(f"  surge_weight={cfg['surge_weight']}", end="")
        if "alpha" in cfg:
            print(f"  alpha={cfg['alpha']}", end="")
        print()
        rows = run_experiment(exp_name, cfg, X_tr, y_tr, X_va, y_va, X_te, y_te)
        all_rows.extend(rows)

    res = pd.DataFrame(all_rows)
    out_path = str(ROOT / "outputs" / "growth_surge_compare.csv")
    res.to_csv(out_path, index=False)
    print(f"\n전체 결과 저장: {out_path}")

    # test set 비교표 출력
    test_res = res[res["split"] == "test"].copy()

    print(f"\n{'='*95}")
    print("=== test set 비교 ===")
    print(f"{'실험':<22} {'MAE':>8} {'RMSE':>8} {'WtdMAE':>9} {'shrink':>9} {'stable':>9} {'surge':>9} {'PB50':>8} {'PB75':>8} {'PB90':>8}")
    print("-"*95)
    for _, row in test_res.iterrows():
        pb50 = f"{row['pinball_50']:.4f}" if row['pinball_50'] is not None else "   -  "
        pb75 = f"{row['pinball_75']:.4f}" if row['pinball_75'] is not None else "   -  "
        pb90 = f"{row['pinball_90']:.4f}" if row['pinball_90'] is not None else "   -  "
        print(
            f"{row['exp']:<22} {row['mae']:>8.4f} {row['rmse']:>8.4f} "
            f"{row['weighted_mae']:>9.4f} {row['mae_shrink']:>9.4f} "
            f"{row['mae_stable']:>9.4f} {row['mae_surge']:>9.4f} "
            f"{pb50:>8} {pb75:>8} {pb90:>8}"
        )

    print(f"\n{'='*95}")
    print("=== valid set 비교 ===")
    valid_res = res[res["split"] == "valid"].copy()
    print(f"{'실험':<22} {'MAE':>8} {'RMSE':>8} {'WtdMAE':>9} {'surge':>9}")
    print("-"*55)
    for _, row in valid_res.iterrows():
        print(f"{row['exp']:<22} {row['mae']:>8.4f} {row['rmse']:>8.4f} {row['weighted_mae']:>9.4f} {row['mae_surge']:>9.4f}")


if __name__ == "__main__":
    main()
