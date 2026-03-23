"""
LightGBM 바이럴 예측 모델
- virality_score : (future_peak - baseline_90d) / (std_90d + 1e-6)
- peak_timing    : argmax(search_trend[t+1:t+30]) + 1  (1~30일)

사용법:
    python modeling/train.py --data data_processed/features.csv
"""

import argparse
import os
import sys
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# feature_engineering/feature_config.py 에서 FEATURE_COLS import
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from feature_engineering.feature_config import FEATURE_COLS

warnings.filterwarnings("ignore")

# ── 상수 ──────────────────────────────────────────────────────────────────────
TARGET_VIRALITY = "virality_score"
TARGET_TIMING   = "peak_timing"
DATE_COL        = "date"

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
# TEST_RATIO  = 0.15

DEFAULT_PARAMS = {
    "boosting_type"    : "gbdt",
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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


# ── 지표 ──────────────────────────────────────────────────────────────────────
def rmse_score(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate(y_true, y_pred, split: str, target: str) -> dict:
    rmse = rmse_score(y_true, y_pred)
    mae  = float(mean_absolute_error(y_true, y_pred))
    print(f"  [{split:5s}] {target}  RMSE={rmse:.4f}  MAE={mae:.4f}")
    return {"split": split, "target": target, "rmse": rmse, "mae": mae}


# ── 데이터 분할 ───────────────────────────────────────────────────────────────
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


# ── LightGBM 학습 ─────────────────────────────────────────────────────────────
def train_lgbm(
    X_train, y_train,
    X_valid, y_valid,
    target: str,
    params: dict,
) -> lgb.LGBMRegressor:

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=200),
        ],
    )
    print(f"  최적 트리 수: {model.best_iteration_}  ({target})")
    return model


# ── SHAP 분석 ─────────────────────────────────────────────────────────────────
def shap_analysis(model: lgb.LGBMRegressor, X: pd.DataFrame, target: str):
    print(f"\nSHAP 분석: {target}")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # bar plot
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f"SHAP Importance (bar) — {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_bar_{target}.png"), dpi=150)
    plt.close()

    # beeswarm plot
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f"SHAP Beeswarm — {target}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"shap_beeswarm_{target}.png"), dpi=150)
    plt.close()

    # 중요도 CSV
    imp_df = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(shap_values).mean(0)})
        .sort_values("mean_abs_shap", ascending=False)
    )
    imp_df.to_csv(os.path.join(OUTPUT_DIR, f"shap_importance_{target}.csv"), index=False)
    print(f"  저장 완료 → {OUTPUT_DIR}/shap_*_{target}.*")


# ── 모델 파이프라인 ────────────────────────────────────────────────────────────
def run_model(
    X_train, y_train,
    X_valid, y_valid,
    X_test,  y_test,
    target: str,
    params: dict,
) -> list[dict]:

    print(f"\n{'='*55}")
    print(f"  {target} 모델")
    print(f"{'='*55}")

    model = train_lgbm(X_train, y_train, X_valid, y_valid, target, params)

    results = []
    for split, X, y in [
        ("train", X_train, y_train),
        ("valid", X_valid, y_valid),
        ("test",  X_test,  y_test),
    ]:
        results.append(evaluate(y, model.predict(X), split, target))

    shap_analysis(model, X_test, target)
    model.booster_.save_model(os.path.join(OUTPUT_DIR, f"lgbm_{target}.txt"))

    return results


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main(csv_path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 데이터 로드
    df = pd.read_csv(csv_path)
    print(f"데이터 로드: {df.shape}  ({csv_path})")

    for col in [TARGET_VIRALITY, TARGET_TIMING] + FEATURE_COLS:
        if col not in df.columns:
            raise ValueError(f"컬럼 '{col}'이 CSV에 없습니다.")

    # 2. 분할
    train_df, valid_df, test_df = split_data(df)
    print(f"train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")
    print(f"feature 수: {len(FEATURE_COLS)}")

    X_train = train_df[FEATURE_COLS]
    X_valid = valid_df[FEATURE_COLS]
    X_test  = test_df[FEATURE_COLS]

    # 3. virality_score 모델
    results = run_model(
        X_train, train_df[TARGET_VIRALITY],
        X_valid, valid_df[TARGET_VIRALITY],
        X_test,  test_df[TARGET_VIRALITY],
        target = TARGET_VIRALITY,
        params = {**DEFAULT_PARAMS},
    )

    # 4. peak_timing 모델 (MAE 직접 최적화)
    results += run_model(
        X_train, train_df[TARGET_TIMING],
        X_valid, valid_df[TARGET_TIMING],
        X_test,  test_df[TARGET_TIMING],
        target = TARGET_TIMING,
        params = {**DEFAULT_PARAMS, "objective": "regression_l1"},
    )

    # 5. 결과 저장
    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(OUTPUT_DIR, "evaluation_results.csv"), index=False)

    print(f"\n{'='*55}")
    print("최종 평가 결과")
    print("="*55)
    print(res_df.to_string(index=False))
    print(f"\n모든 결과 저장 완료 → {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="data_processed/features.csv",
        help="feature + target 컬럼이 포함된 CSV 경로",
    )
    args = parser.parse_args()
    main(args.data)
