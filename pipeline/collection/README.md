# 키워드 수집 파이프라인

## 개요

네이버 데이터랩에서 식품 > 과자/베이커리 키워드 및 트렌드 데이터 수집

1. 쇼핑인사이트 내부 API로 월별 키워드 Top 500 크롤링 (Playwright 세션, UI 조작 없음)
2. 계층 샘플링으로 500개 키워드 선정
3. 오픈API로 선정 키워드의 일별 쇼핑 클릭 트렌드 + 통합 검색 트렌드 수집

## 파일 구조

```
pipeline/collection/
├── crawl_keywords.py              # 1단계: API 호출 → 윈도우별 CSV 저장 + 검증·통합
├── select_keywords.py             # 2단계: 통합 CSV에서 계층 샘플링으로 500개 선정
├── crawl_shopping_click_trend.py  # 3단계: 선정 키워드 일별 쇼핑 클릭 트렌드 수집
├── crawl_search_trend.py          # 4단계: 선정 키워드 일별 통합 검색 트렌드 수집
└── README.md
```

## 실행 방법

```bash
python pipeline/collection/crawl_keywords.py
python pipeline/collection/select_keywords.py
python pipeline/collection/crawl_shopping_click_trend.py
python pipeline/collection/crawl_search_trend.py
```

## 산출물

```
data/raw/
├── keyword_pool/                                             # 키워드 크롤링 산출물
│   ├── windows/                                              #   윈도우별 중간산출물
│   │   └── kw_식품_과자베이커리_*.csv                         #     50개 월별 CSV
│   ├── keywords_all_식품_과자베이커리.csv                     #   전체 통합 (25,000행)
│   └── keywords_50_seed{시드}.csv                            #   테스트용 50개 샘플
├── keyword_select/                                           # 키워드 선정 산출물
│   └── keywords_selected_500_seed{시드}.csv                  #   최종 선정 500개
├── shopping_click/                                           # 쇼핑 클릭 트렌드 산출물
│   └── trend.csv                                             #   키워드×일 쇼핑 클릭 트렌드
└── search/                                                   # 통합 검색 트렌드 산출물
    └── trend.csv                                             #   키워드×일 통합 검색 트렌드
```

### CSV 네이밍 규칙

`kw_{1분류}_{2분류}_{라벨}_{시작YYYYMMDD}_{종료YYYYMMDD}.csv`

### 윈도우별 CSV 컬럼

| 컬럼 | 설명 |
|------|------|
| rank | 해당 윈도우 내 순위 (1~500) |
| keyword | 키워드명 |
| window | 윈도우 라벨 (2022-01) |
| window_start | 시작일 (2022-01-01) |
| window_end | 종료일 (2022-01-31) |

### 최종 선정 CSV 컬럼

| 컬럼 | 설명 |
|------|------|
| keyword | 키워드명 |
| stratum | 사분위 층 (Q1_rare / Q2_low / Q3_mid / Q4_freq) |
| appearance | 출현 윈도우 수 |
| best_rank | 최고 순위 |
| rank_std | 순위 변동성 (표준편차) |
| first_seen | 첫 등장 윈도우 |
| last_seen | 마지막 등장 윈도우 |

### 트렌드 CSV 구조 (쇼핑 클릭 / 검색 공통)

행이 키워드, 열이 날짜(2022-01-01 ~ 2026-02-28)인 피벗 형태
값은 네이버 데이터랩 오픈API가 반환하는 상대 비율(0~100, 기간 내 최대값=100)
키워드 존재 기간 외의 날짜는 NaN
- `shopping_click/trend.csv`: 네이버 쇼핑 내 상품 클릭량 기반
- `search/trend.csv`: 네이버 통합 검색량 기반

## 그룹 분류

50개 윈도우 통합 시 고유 키워드 3,576개
이 중 4개월 이상 등장한 1,303개를 대상으로, `appearance` 사분위수(`pd.qcut`)로 4개 층을 나눈 뒤 층마다 균등 추출
바이럴 예측 모델에 바이럴/비바이럴 키워드가 모두 필요하므로, 출현 빈도별로 골고루 섞이도록 구성

| 층 | appearance 범위 | 의미 | 선정 |
|------|-----------------|------|------|
| Q1_rare | 4~5 | 거의 등장하지 않은 키워드 | 125개 |
| Q2_low | 6~10 | 가끔 등장 | 125개 |
| Q3_mid | 11~23 | 중간 빈도 | 125개 |
| Q4_freq | 24~50 | 스테디셀러 | 125개 |

## 설정

모든 설정은 `configs/collection.yaml`에서 관리

```yaml
crawl:
  cid_1st: "50000006"       # 식품
  cid_2nd: "50000149"       # 과자/베이커리
  start_date: "2022-01-01"
  end_date: "2026-03-01"
  window_months: 1
  pages: 25                  # 20개 × 25 = 500

select:
  final_n: 500
  min_appearance: 4
  seed: null                 # null이면 매 실행마다 랜덤 시드
  strata:
    Q1_rare: 125
    Q2_low: 125
    Q3_mid: 125
    Q4_freq: 125

click_trend:
  api_url: "https://openapi.naver.com/v1/datalab/shopping/category/keywords"
  category: "50000149"
  start_date: "2022-01-01"
  end_date: "2026-02-28"
  time_unit: "date"
  sleep_seconds: 0.5
  input_csv: "keywords_selected_500_seed1527267025.csv"

search_trend:
  api_url: "https://openapi.naver.com/v1/datalab/search"
  start_date: "2022-01-01"
  end_date: "2026-02-28"
  time_unit: "date"
  sleep_seconds: 0.5
  input_csv: "keywords_selected_500_seed1527267025.csv"
```

`crawl_shopping_click_trend.py`, `crawl_search_trend.py`는 `.env`에서 `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET`을 읽어 오픈API 인증에 사용

## 재실행 안전성

`crawl_keywords.py`는 이미 저장된 CSV를 확인하고 스킵
중단 후 재실행하면 남은 윈도우만 이어서 크롤링
