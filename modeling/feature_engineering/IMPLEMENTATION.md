# v2 피처 구현 완료 보고서

## 개요

v2 피처 테이블 (107개)의 전체 구현이 완료되었습니다. 문서 명세대로 모든 피처가 정의되고 계산 로직이 구현되었습니다.

---

## 구현 현황

### 1. 검색+클릭 피처 (60개)
**파일**: `modeling/feature_engineering/search_features.py`

- **검색량 기반**: 27개
  - 이동평균 (5개): 3d, 5d, 7d, 14d, 30d
  - 성장률 (5개): 3d, 5d, 7d, 14d, 30d
  - 기울기 (5개): 3d, 5d, 7d, 14d, 30d
  - 변동성 (3개): 7d, 14d, 30d
  - 위치 (3개): 7d, 14d, 30d
  - 최고점 경과일 (3개): 7d, 14d, 30d
  - 가속도 (3개): short, mid, long

- **클릭수 기반**: 27개 (동일 구조, `click_*` 접두사)

- **검색+클릭 결합**: 6개
  - `click_search_ratio`: 검색 대비 클릭 전환율 (7일)
  - `click_lead_7d`, `click_lead_30d`: 클릭이 검색을 앞서는 정도
  - `engage_force_7d`, `engage_force_30d`: 클릭 강도 × 검색 모멘텀
  - `click_peak_gap`: 14일 최고점 위치 차이

**함수**: `compute_all_features(df: DataFrame) -> DataFrame`
- 입력: keyword, date, search, click 컬럼
- 출력: 60개 피처 추가

---

### 2. 언급량 피처 (33개)
**파일**: `modeling/feature_engineering/mention_features.py`

- **언급량 기반**: 27개 (검색과 동일 구조, `mention_*` 접두사)

- **검색+언급 결합**: 6개
  - `mention_search_ratio`: 검색 대비 SNS 언급 비율
  - `mention_lead_7d`, `mention_lead_30d`: 언급이 검색을 앞서는 정도
  - `viral_force_7d`, `viral_force_30d`: 언급 강도 × 검색 모멘텀
  - `mention_peak_gap`: 14일 최고점 위치 차이

**함수**: `compute_mention_features(df: DataFrame) -> DataFrame`
- 입력: keyword, date, search, mention 컬럼
- 출력: 33개 피처 추가

---

### 3. 인구통계 피처 (14개)
**파일**: `modeling/feature_engineering/demographic_features.py`

- **성별 피처** (4개):
  - `male_click_ratio`, `female_click_ratio`: 성별 비율 (t-1)
  - `gender_click_skew`: 성별 편중도
  - `gender_click_shift_7d`: 7일 남성 비율 변화

- **연령대 피처** (10개):
  - 개별 연령대: age10, age20, age30, age40, age50p_click_ratio
  - 그룹별: young (10+20대), mid (30+40대), core (최대값)
  - `age_click_entropy`: 연령 분포 엔트로피
  - `age_click_shift_7d`: 핵심 연령대 7일 변화

**함수**: `compute_demographic_features(df: DataFrame) -> DataFrame`
- 입력: keyword, date, gender_m_pct, gender_f_pct, age_*_pct 컬럼
- 출력: 14개 피처 추가

---

### 4. 피처 설정 정의
**파일**: `modeling/feature_engineering/feature_config.py`

**주요 변수**:
- `META_COLS`: [keyword, date]
- `RAW_COLS`: [search, click, mention]
- `FEATURE_COLS`: 107개 모든 피처 (canonical 순서)
- `TARGET_COLS`: [virality_score, peak_time]
- `ALL_COLS`: 위 네 그룹의 합계 (127개)
- `REQUIRED_COLS`: FEATURE_COLS + TARGET_COLS (학습에 필수)

---

### 5. 통합 피처 계산 함수
**파일**: `modeling/feature_engineering/compute_features.py`

**함수**: `compute_all_features_unified(df: DataFrame) -> DataFrame`
- 입력: keyword, date, search, click, mention + demographic 컬럼
- 처리:
  1. `compute_all_features()` 호출 (search + click 60개)
  2. `compute_mention_features()` 호출 (mention 33개)
  3. `compute_demographic_features()` 호출 (demographic 14개)
  4. 모든 피처를 `FEATURE_COLS` 순서대로 정렬
- 출력: 127개 컬럼 (메타 3 + raw 3 + 피처 107 + 나머지)

**사용법**:
```python
from modeling.feature_engineering.compute_features import compute_all_features_unified

df_raw = pd.DataFrame({
    'keyword': [...],
    'date': [...],
    'search': [...],
    'click': [...],
    'mention': [...],
    'gender_m_pct': [...],
    'gender_f_pct': [...],
    'age_10_pct': [...],
    # ... 나머지 demographic 컬럼
})

df_features = compute_all_features_unified(df_raw)
# → df_features는 모든 107개 피처를 포함한 DataFrame
```

---

## 구현 검증

### 피처 수 확인
```python
from modeling.feature_engineering.feature_config import FEATURE_COLS

len(FEATURE_COLS)  # 107개 확인
```

| 카테고리 | 개수 | 비고 |
|---------|------|------|
| 검색량 | 27 | ma(5) + growth(5) + slope(5) + std(3) + pos(3) + days(3) + accel(3) |
| 클릭수 | 27 | 동일 구조 |
| 검색+클릭 | 6 | ratio + lead×2 + force×2 + gap |
| 언급량 | 27 | 동일 구조 |
| 검색+언급 | 6 | 동일 구조 |
| 인구통계 | 14 | 성별(4) + 연령(10) |
| **합계** | **107** | ✓ |

### 계산식 검증
모든 계산식은 [docs/피처_정의_v2.md](../../docs/피처_정의_v2.md)의 명세와 일치합니다.

- ε (epsilon) = 1e-6 (0 나눗셈 방지)
- time window: 3d, 5d, 7d, 14d, 30d
- lookback: 최대 60일
- t-1 지표: demographic 피처 (전일 데이터 사용)

---

## 데이터 파이프라인 요구사항

### 필수 컬럼 (입력)

| 컬럼명 | 타입 | 출처 | 설명 |
|--------|------|------|------|
| keyword | str | 메타 | 키워드 식별자 |
| date | datetime | 메타 | 수집 일자 |
| search | float | 검색 신호 | 일별 검색량 (절대값) |
| click | float | 검색 신호 | 일별 클릭수 (절대값) |
| mention | float | SNS 신호 | 일별 언급량 |
| gender_m_pct | float | 쇼핑탭 | 남성 클릭 비율 |
| gender_f_pct | float | 쇼핑탭 | 여성 클릭 비율 |
| age_10_pct | float | 쇼핑탭 | 10대 클릭 비율 |
| age_20_pct | float | 쇼핑탭 | 20대 클릭 비율 |
| age_30_pct | float | 쇼핑탭 | 30대 클릭 비율 |
| age_40_pct | float | 쇼핑탭 | 40대 클릭 비율 |
| age_50_pct | float | 쇼핑탭 | 50대 클릭 비율 (`age_50_pct + age_60_pct`) |
| age_60_pct | float | 쇼핑탭 | 60대 이상 클릭 비율 (위에 통합) |

위 컬럼들이 모두 제공되면 `compute_all_features_unified()`로 107개 피처를 자동 계산할 수 있습니다.

---

## 파일 구조

```
modeling/feature_engineering/
├── __init__.py
├── feature_config.py              # 피처 메타데이터 (107개 + 스키마 정의)
├── search_features.py             # 검색/클릭 피처 (60개)
├── mention_features.py            # 언급 피처 (33개)
├── demographic_features.py        # 인구통계 피처 (14개)
├── compute_features.py            # 통합 함수
└── README.md                       # 이 문서
```

---

## 다음 단계

1. **데이터 파이프라인 확인**: 
   - `pipeline/` 폴더에서 mention 및 demographic 컬럼이 생성되는지 확인
   - 필요하면 파이프라인 코드 일부 수정

2. **데이터셋 재구성**:
   - 기존 `data/processed/dataset_final.csv` 대신 새 v2 데이터셋 생성
   - `compute_all_features_unified()`를 통해 모든 107개 피처 계산

3. **모델 재학습**:
   - 새 feature_config.FEATURE_COLS 기준으로 모델 재학습
   - SHAP/피처 중요도 분석 업데이트

---

## 참고

- **구현 완료일**: 2026년 04월 02일
- **문서**: [docs/피처_정의_v2.md](../../docs/피처_정의_v2.md)
- **이전 버전 매핑**: [docs/피처_정의_v2.md#v1-피처명-대응표](../../docs/피처_정의_v2.md#v1-피처명-대응표)
