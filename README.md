# capstone-viral-product-ml

과자/베이커리 키워드의 바이럴 강도 예측 ML 프로젝트 (캡스톤 8조)

## 디렉토리 구조

```
crawling/                # 데이터 수집 스크립트
feature_engineering/     # 피처 엔지니어링 모듈
modeling/
  train.py               # LightGBM 학습 파이프라인 (Optuna TPE 튜닝 포함)
  data_loader.py         # CSV 로드 및 컬럼 검증
  outputs/               # 학습 결과 (모델, SHAP, 평가지표)
data_raw/                # 원본 데이터 (dataset_final.csv)
data_processed/          # 전처리된 데이터
```

## 피처 구성 (41개)

| 그룹 | 피처 수 | 설명 |
|------|--------|------|
| 검색량 기반 | 10 | 이동평균, 성장률, 기울기, 가속도, 변동성 등 |
| 클릭수 기반 | 10 | 검색량 기반과 동일한 구조 |
| 검색+클릭 결합 | 4 | 클릭/검색 비율, 선행 신호, 인게이지먼트 강도 등 |
| 언급량 기반 | 9 | 언급량 이동평균, 성장률, 가속도 등 |
| 검색-언급 결합 | 4 | 언급/검색 비율, 선행 신호, 바이럴 강도 등 |
| 클릭 비율 | 4 | 성별·연령대별 클릭 비율, 엔트로피 등 |

## 타겟

- `virality_score` : `(future_peak - baseline_90d) / (std_90d + 1e-6)`
- `peak_time` : argmax(search_trend[t+1:t+30]) + 1  (1~30일)

## 사용법

```bash
# 기본 파라미터로 학습
python modeling/train.py --data data_raw/dataset_final.csv

# Optuna TPE 튜닝 후 학습 (trial 50회)
python modeling/train.py --data data_raw/dataset_final.csv --tune --n_trials 50
```

## 학습 결과 (튜닝 전, 기본 파라미터)

### virality_score

| split | RMSE | MAE |
|-------|------|-----|
| train | 9.186 | 0.954 |
| valid | 2.677 | 0.859 |
| test  | 2.486 | 0.898 |

**상위 피처 (SHAP):** search_growth_14d, search_ma_7d, search_growth_7d, search_acceleration, click_growth_14d

### peak_time

| split | RMSE | MAE |
|-------|------|-----|
| train | 4.194 | 3.625 |
| valid | 4.252 | 3.688 |
| test  | 4.312 | 3.757 |

**상위 피처 (SHAP):** search_std_7d, search_acceleration, click_ma_7d, search_growth_14d, search_slope_14d

> 데이터: `data_raw/dataset_final.csv` (573,285행)
> 분할: train 70% / valid 15% / test 15% (시계열 순서 기반)
