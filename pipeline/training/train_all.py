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
    "buzz_composite_5d":  "staged3", "buzz_composite_10d":  "staged3", "buzz_composite_15d":  "staged3",
    "sustainability_5d":  "single",  "sustainability_10d":  "single",  "sustainability_15d":  "single",
}

# 구간 설정 — 패밀리별 경계(원본 스케일), 구간명, 회귀 시 log1p 여부
BUCKET_CONFIG = {
    "intensity":     {"thresholds": [569.0, 800.0],       "names": ["low",    "mid",    "high"],                  "log": True,  "n_class": 3, "hard": False},
    "growth":        {"thresholds": [0.90,  1.56,  5.0],  "names": ["shrink", "stable", "surge", "extreme"],      "log": True,  "n_class": 4, "hard": False},
    "buzz_composite":  {"thresholds": [-0.50, 0.80],        "names": ["negative", "neutral", "positive"],           "log": False, "n_class": 3, "hard": False, "hierarchical": True},
    "sustainability":  {"thresholds": [0.80,  1.20],        "names": ["declining", "stable",   "sustained"],        "log": True,  "n_class": 3, "hard": False},
}

# 구간별 회귀 파라미터 override — 패밀리+구간명 키로 DEFAULT_REG_PARAMS 일부를 덮어씀
# alphas 키가 있으면 해당 alpha 목록으로 quantile 앙상블 학습
BUCKET_REG_OVERRIDE = {
    ("intensity", "high"):         {"num_leaves": 32, "min_child_samples": 50},
    ("growth", "shrink"):          {"num_leaves": 32, "min_child_samples": 30},
    ("growth", "stable"):          {"num_leaves": 32, "min_child_samples": 30},
    ("growth", "surge"):           {"objective": "quantile", "num_leaves": 64, "min_child_samples": 20, "alphas": [0.75, 0.90]},
    ("growth", "extreme"):         {"objective": "quantile", "num_leaves": 16, "min_child_samples": 10, "alphas": [0.75, 0.90]},
    ("buzz_composite", "positive"): {"log": True, "objective": "regression_l1"},
}

# 구간별 Optuna 탐색 범위 — (nl_low, nl_high, md_low, md_high, mc_low, mc_high)
# 미정의 구간은 기본값(15, 127, 4, 10, 20, 200) 사용
TUNE_SEARCH_SPACE = {
    ("buzz_composite", "clf"):      (31, 127, 4, 10,  20, 200),
    ("buzz_composite", "clf1"):     (15,  63, 4,  8,  20, 200),  # neg vs rest
    ("buzz_composite", "clf2"):     (31, 127, 4, 10,  20, 200),  # neutral vs positive
    ("buzz_composite", "negative"): (15,  63, 4,  8,  20, 200),
    ("buzz_composite", "neutral"):  (31, 127, 4, 10,  20, 200),
    ("buzz_composite", "positive"): (15,  47, 4,  8,  10, 100),
}

# extreme 구간(양 끝) 샘플 가중치 — 소수 클래스 학습 보완
EXTREME_WEIGHT = 2.0

# sustainability 전용 파생 피처 — 기존 FEAT_COLS에서 계산, 데이터셋 재빌드 없이 추가
SUST_EXTRA_FEATS = [
    "search_stability", "search_pos_trend", "search_peak_retention",
    "search_slope_ratio", "search_trend_quality", "search_vol_regime",
]


def _add_sust_features(df):
    # search_cv_14d 역수 — 변동성이 낮을수록 지속성 높음
    df["search_stability"]      = (1.0 / (df["search_cv_14d"] + 1e-6)).clip(0, 50)
    # 단기 vs 중기 포지션 차이 — 포지션 상승/하락 방향
    df["search_pos_trend"]      = df["search_pos_7d"] - df["search_pos_14d"]
    # 현재 포지션 대비 30일 고점 유지율 — 얼마나 버티는지
    df["search_peak_retention"] = (df["search_pos_7d"] / (df["search_pos_30d"] + 1e-6)).clip(0, 5)
    # 단기/중기 기울기 비 — 모멘텀 가속 여부 (>1: 최근 강화, <1: 감속)
    df["search_slope_ratio"]    = (df["search_slope_7d"] / (df["search_slope_14d"] + 1e-6)).clip(-5, 5)
    # 방향성 있는 트렌드 강도 — trend_r2가 높고 상승 중이면 양수
    df["search_trend_quality"]  = df["search_trend_r2_7d"] * np.sign(df["search_slope_7d"])
    # 중기 대비 단기 변동성 변화 — 음수면 안정화 중
    df["search_vol_regime"]     = (df["search_cv_14d"] - df["search_cv_30d"]).clip(-5, 5)
    return df

# 타겟별 single 회귀 파라미터 override — DEFAULT_PARAMS 일부를 덮어씀
SINGLE_PARAMS_OVERRIDE = {
    "sustainability_5d":  {"learning_rate": 0.01, "n_estimators": 3000, "device_type": "cuda"},
    "sustainability_10d": {"learning_rate": 0.01, "n_estimators": 3000, "device_type": "cuda"},
    "sustainability_15d": {"learning_rate": 0.01, "n_estimators": 3000, "device_type": "cuda"},
}

DEFAULT_CLF_PARAMS = {
    "boosting_type": "gbdt", "objective": "multiclass",
    "n_estimators": 3000, "learning_rate": 0.01, "num_leaves": 64, "max_depth": -1,
    "min_child_samples": 200, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "random_state": 42, "n_jobs": -1, "verbose": -1,
    "metric": "multi_logloss",
}

DEFAULT_REG_PARAMS = {
    "boosting_type": "gbdt", "objective": "regression",
    "n_estimators": 3000, "learning_rate": 0.01, "num_leaves": 64, "max_depth": -1,
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


def soft_predict(clf, regressors, fallbacks, X, log_flags, hard=False):
    # clf 확률 가중합(soft) 또는 argmax 라우팅(hard)으로 최종 예측
    # clf가 tuple이면 계층적 이진 분류기: (clf1=neg vs rest, clf2=neutral vs pos)
    # log_flags: bucket index → log1p 적용 여부 (per-bucket 독립 적용)
    # regressors 값이 list이면 앙상블(평균), 단일 모델이면 그대로 사용
    if isinstance(clf, tuple):
        if len(clf) == 2:
            clf1, clf2 = clf
            p_nn  = clf1.predict_proba(X)[:, 1]
            p_pos = clf2.predict_proba(X)[:, 1]
            proba = np.stack([
                1.0 - p_nn,
                p_nn * (1.0 - p_pos),
                p_nn * p_pos,
            ], axis=1)
        else:  # 3-stage: neg vs rest → neutral vs (mild+strong) → mild vs strong
            clf1, clf2, clf3 = clf
            p_nn     = clf1.predict_proba(X)[:, 1]
            p_pos    = clf2.predict_proba(X)[:, 1]
            p_strong = clf3.predict_proba(X)[:, 1]
            proba = np.stack([
                1.0 - p_nn,
                p_nn * (1.0 - p_pos),
                p_nn * p_pos * (1.0 - p_strong),
                p_nn * p_pos * p_strong,
            ], axis=1)
    else:
        proba = clf.predict_proba(X)
    buckets = np.argmax(proba, axis=1)  # hard routing용 구간 인덱스

    def _reg_pred(b_idx, model):
        use_log = log_flags.get(b_idx, False)
        if model is None:
            return np.full(len(X), fallbacks[b_idx])
        elif isinstance(model, list):
            raw = np.stack([m.predict(X) for m in model], axis=0).mean(axis=0)
            return np.expm1(raw) if use_log else raw
        else:
            raw = model.predict(X)
            return np.expm1(raw) if use_log else raw

    if hard:
        preds = np.zeros(len(X))
        for b_idx, model in regressors.items():
            mask = buckets == b_idx
            if mask.sum() == 0:
                continue
            full_pred      = _reg_pred(b_idx, model)
            preds[mask]    = full_pred[mask]
        return preds

    preds = np.zeros(len(X))
    for b_idx, model in regressors.items():
        preds += proba[:, b_idx] * _reg_pred(b_idx, model)
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


def _run_study(X_tr, y_tr, X_va, y_va, fixed, n_trials, w_tr, is_clf, label, dirs,
               nl_low=15, nl_high=127, md_low=4, md_high=10, mc_low=20, mc_high=200):
    # Optuna study — num_leaves/max_depth/min_child_samples 탐색, trial별 best_iteration·과적합 기록
    trial_records = []

    def objective(trial):
        params = {
            **fixed,
            "n_estimators":      500,
            "num_leaves":        trial.suggest_int("num_leaves",        nl_low, nl_high),
            "max_depth":         trial.suggest_int("max_depth",         md_low, md_high),
            "min_child_samples": trial.suggest_int("min_child_samples", mc_low, mc_high),
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
            "trial":             trial.number,
            "num_leaves":        params["num_leaves"],
            "max_depth":         params["max_depth"],
            "min_child_samples": params["min_child_samples"],
            "learning_rate":     params["learning_rate"],
            "best_iteration":    best_iter,
            "train_score":       round(score_tr, 6),
            "valid_score":       round(score_va, 6),
            "overfit_gap":       overfit_gap,
        })
        return score_va

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    (pd.DataFrame(trial_records)
     .sort_values("valid_score")
     .reset_index(drop=True)
     .to_csv(os.path.join(dirs["metrics"], f"tune_trials_{label}.csv"), index=False))

    return study.best_params, study.best_value


def _tune(X_tr, y_tr, X_va, y_va, fixed, n_trials, dirs, w_tr=None, label="",
          nl_low=15, nl_high=127, md_low=4, md_high=10, mc_low=20, mc_high=200):
    # num_leaves / max_depth / min_child_samples Optuna 탐색 — learning_rate는 DEFAULT_PARAMS에서 수동 설정
    is_clf = fixed.get("objective") in ("multiclass", "binary")
    lr     = fixed.get("learning_rate", 0.05)
    print(f"    [tune] {label}  lr={lr}  trials={n_trials}  nl=[{nl_low},{nl_high}]  md=[{md_low},{md_high}]  mc=[{mc_low},{mc_high}]")

    best, val = _run_study(X_tr, y_tr, X_va, y_va, fixed, n_trials, w_tr, is_clf, label, dirs,
                           nl_low=nl_low, nl_high=nl_high, md_low=md_low, md_high=md_high,
                           mc_low=mc_low, mc_high=mc_high)
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
    print(f"  [{split}] MAE={mae:.4f}  RMSE={rmse:.4f}")
    return {"target": target, "split": split, "mae": mae, "rmse": rmse, "n": len(y_true)}


def run_staged(X_tr, y_tr, X_va, y_va, X_te, y_te, target, dirs, do_tune, n_trials):
    fam        = _family(target)
    cfg        = BUCKET_CONFIG[fam]
    thresholds = cfg["thresholds"]
    names      = cfg["names"]
    use_log    = cfg["log"]
    n_class    = cfg["n_class"]
    hard         = cfg.get("hard", False)
    hierarchical = cfg.get("hierarchical", False)

    print(f"\n{'='*55}")
    print(f"  [staged] {target}  thresholds={thresholds}  log={use_log}  n_class={n_class}  hard={hard}  hierarchical={hierarchical}")
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
    if hierarchical:
        # clf1: negative(0) vs (neutral+positive)(1)
        b_bin1_tr = (b_tr > 0).astype(int)
        b_bin1_va = (b_va > 0).astype(int)
        w_clf1    = np.where(b_tr == 0, EXTREME_WEIGHT, 1.0)
        clf1_params = {**DEFAULT_CLF_PARAMS, "objective": "binary", "metric": "binary_logloss"}
        if do_tune:
            nl_low, nl_high, md_low, md_high, mc_low, mc_high = TUNE_SEARCH_SPACE.get((fam, "clf1"), (15, 127, 4, 10, 20, 200))
            clf1_params = _tune(X_tr, b_bin1_tr, X_va, b_bin1_va, clf1_params, n_trials, dirs, w_clf1,
                                label=f"{target}_clf1", nl_low=nl_low, nl_high=nl_high,
                                md_low=md_low, md_high=md_high, mc_low=mc_low, mc_high=mc_high)
        clf1 = lgb.LGBMClassifier(**clf1_params)
        clf1.fit(X_tr, b_bin1_tr, sample_weight=w_clf1, eval_set=[(X_va, b_bin1_va)],
                 callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        print(f"    clf1 (neg vs rest) valid accuracy: {float(np.mean(clf1.predict(X_va) == b_bin1_va)):.4f}")
        clf1.booster_.save_model(os.path.join(dirs["models"], f"lgbm_{target}_clf1.txt"))
        _shap_save(clf1, X_te.iloc[:min(3000, len(X_te))], f"{target}_clf1", dirs)

        # clf2: neutral(0) vs positive(1) — non-negative 샘플만 사용
        mask_non_neg_tr = b_tr > 0
        mask_non_neg_va = b_va > 0
        b_bin2_tr = (b_tr[mask_non_neg_tr] == 2).astype(int)
        b_bin2_va = (b_va[mask_non_neg_va] == 2).astype(int)
        w_clf2    = np.where(b_bin2_tr == 1, EXTREME_WEIGHT, 1.0)
        clf2_params = {**DEFAULT_CLF_PARAMS, "objective": "binary", "metric": "binary_logloss"}
        if do_tune:
            nl_low, nl_high, md_low, md_high, mc_low, mc_high = TUNE_SEARCH_SPACE.get((fam, "clf2"), (31, 127, 4, 10, 20, 200))
            clf2_params = _tune(X_tr[mask_non_neg_tr], b_bin2_tr, X_va[mask_non_neg_va], b_bin2_va,
                                clf2_params, n_trials, dirs, w_clf2,
                                label=f"{target}_clf2", nl_low=nl_low, nl_high=nl_high,
                                md_low=md_low, md_high=md_high, mc_low=mc_low, mc_high=mc_high)
        clf2 = lgb.LGBMClassifier(**clf2_params)
        clf2.fit(X_tr[mask_non_neg_tr], b_bin2_tr, sample_weight=w_clf2,
                 eval_set=[(X_va[mask_non_neg_va], b_bin2_va)],
                 callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        print(f"    clf2 (neutral vs pos) valid accuracy: {float(np.mean(clf2.predict(X_va[mask_non_neg_va]) == b_bin2_va)):.4f}")
        clf2.booster_.save_model(os.path.join(dirs["models"], f"lgbm_{target}_clf2.txt"))
        _shap_save(clf2, X_te[b_te > 0].iloc[:min(3000, int((b_te > 0).sum()))], f"{target}_clf2", dirs)

        # 계층적 결합 3-class 정확도
        p_nn  = clf1.predict_proba(X_va)[:, 1]
        p_pos = clf2.predict_proba(X_va)[:, 1]
        proba_combined = np.stack([1 - p_nn, p_nn * (1 - p_pos), p_nn * p_pos], axis=1)
        acc_combined = float(np.mean(np.argmax(proba_combined, axis=1) == b_va))
        print(f"    combined 3-class valid accuracy: {acc_combined:.4f}")
        clf = (clf1, clf2)
    else:
        clf_params = {**DEFAULT_CLF_PARAMS, "num_class": n_class}
        if do_tune:
            nl_low, nl_high, md_low, md_high, mc_low, mc_high = TUNE_SEARCH_SPACE.get((fam, "clf"), (15, 127, 4, 10, 20, 200))
            clf_params = _tune(X_tr, b_tr, X_va, b_va, clf_params, n_trials, dirs, w_tr,
                               label=f"{target}_clf", nl_low=nl_low, nl_high=nl_high,
                               md_low=md_low, md_high=md_high, mc_low=mc_low, mc_high=mc_high)
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
    log_flags  = {}  # bucket index → log1p 적용 여부 — soft_predict에 전달

    for b_idx, name in enumerate(names):
        mask_tr = b_tr == b_idx
        mask_va = b_va == b_idx
        fallbacks[b_idx] = float(np.median(y_tr[mask_tr])) if mask_tr.sum() > 0 else 0.0

        print(f"\n    [{name}] train={mask_tr.sum()}  valid={mask_va.sum()}")
        if mask_tr.sum() < 10:
            regressors[b_idx] = None
            log_flags[b_idx]  = use_log
            continue

        raw_override = BUCKET_REG_OVERRIDE.get((fam, name), {})
        alphas       = raw_override.get("alphas", None)
        # log 키 — 지정 시 해당 구간만 독립 적용, 미지정 시 family 기본값 사용
        bucket_log   = raw_override.get("log", use_log)
        override     = {k: v for k, v in raw_override.items() if k not in ("alphas", "log")}
        reg_params   = {**DEFAULT_REG_PARAMS, **override}
        log_flags[b_idx] = bucket_log

        y_tr_b = np.log1p(y_tr[mask_tr]) if bucket_log else y_tr[mask_tr]

        # valid 샘플 부족 시 train 일부를 대용
        if mask_va.sum() >= 5:
            xv_b = X_va[mask_va]
            yv_b = np.log1p(y_va[mask_va]) if bucket_log else y_va[mask_va]
        else:
            k    = max(1, mask_tr.sum() // 5)
            xv_b = X_tr[mask_tr].iloc[:k]
            yv_b = y_tr_b[:k]

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
                nl_low, nl_high, md_low, md_high, mc_low, mc_high = TUNE_SEARCH_SPACE.get((fam, name), (15, 127, 4, 10, 20, 200))
                reg_params = _tune(X_tr[mask_tr], y_tr_b, xv_b, yv_b,
                                   reg_params, n_trials, dirs, label=f"{target}_reg_{name}",
                                   nl_low=nl_low, nl_high=nl_high, md_low=md_low, md_high=md_high,
                                   mc_low=mc_low, mc_high=mc_high)
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
        pred = soft_predict(clf, regressors, fallbacks, X, log_flags, hard=hard)
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

    params = {**DEFAULT_PARAMS, **SINGLE_PARAMS_OVERRIDE.get(target, {})}
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
    df = _add_sust_features(df)
    print(f"shape: {df.shape}")

    print(f"\n시계열 분할 (gap={GAP_DAYS}일, stride={args.train_stride})")
    train_df, valid_df, test_df = split_data(df, args.train_stride)
    print(f"features: {len(FEAT_COLS)}")

    all_targets = list(TARGET_STRATEGY.keys())
    targets = all_targets if args.targets == "all" else [t.strip() for t in args.targets.split(",")]

    all_results = []
    for target in targets:
        strategy = TARGET_STRATEGY.get(target, "single")
        feat_cols = FEAT_COLS + SUST_EXTRA_FEATS if _family(target) == "sustainability" else FEAT_COLS
        X_train = train_df[feat_cols]
        X_valid = valid_df[feat_cols]
        X_test  = test_df[feat_cols]
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
