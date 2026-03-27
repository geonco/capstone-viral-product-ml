# 키워드 수집 파이프라인

## 개요

네이버 데이터랩에서 식품 > 과자/베이커리 키워드 및 트렌드 데이터 수집

1. 쇼핑인사이트 내부 API로 월별 키워드 Top 500 크롤링 (Playwright 세션, UI 조작 없음)
2. 계층 샘플링으로 500개 키워드 선정
3. 오픈API로 선정 키워드의 일별 쇼핑 클릭 트렌드 + 통합 검색 트렌드 수집
4. 성별·연령별 쇼핑 클릭 비율(%) 산출

## 파일 구조

```
pipeline/collection/
├── crawl_keywords.py              # 1단계: API 호출 → 윈도우별 CSV 저장 + 검증·통합
├── select_keywords.py             # 2단계: 통합 CSV에서 계층 샘플링으로 500개 선정
├── crawl_shopping_click_trend.py   # 3단계: 선정 키워드 일별 쇼핑 클릭 트렌드 수집
├── crawl_search_trend.py           # 4단계: 선정 키워드 일별 통합 검색 트렌드 수집
├── crawl_shopping_click_gender.py  # 5단계: 선정 키워드 일별 성별 쇼핑 클릭 트렌드 수집
├── crawl_shopping_click_age.py     # 6단계: 선정 키워드 일별 연령별 쇼핑 클릭 트렌드 수집
├── derive_gender_proportion.py     # 7단계: 성별 ratio → 남/여 비율(%) 산출
├── derive_age_proportion.py        # 8단계: 연령별 ratio → 연령대 비율(%) 산출
├── ratio_to_absolute.py               # ratio ↔ absolute 변환 공유 모듈
├── estimate_shopping_click_absolute.py  # 9a: 쇼핑 클릭 ratio → 절대값 역산
├── estimate_search_absolute.py         # 9b: 검색 트렌드 ratio → 절대값 역산
├── verify_shopping_click_absolute.py   # 10a: 쇼핑 클릭 절대값 검증 (round-trip)
├── verify_search_absolute.py           # 10b: 검색 절대값 검증 (round-trip + 교차 + 앵커)
└── README.md
```

## 실행 방법

```bash
python pipeline/collection/crawl_keywords.py
python pipeline/collection/select_keywords.py
python pipeline/collection/crawl_shopping_click_trend.py
python pipeline/collection/crawl_search_trend.py
python pipeline/collection/crawl_shopping_click_gender.py
python pipeline/collection/crawl_shopping_click_age.py
python pipeline/collection/derive_gender_proportion.py
python pipeline/collection/derive_age_proportion.py
python pipeline/collection/estimate_shopping_click_absolute.py
python pipeline/collection/estimate_search_absolute.py
python pipeline/collection/verify_shopping_click_absolute.py
python pipeline/collection/verify_search_absolute.py
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
│   ├── trend.csv                                             #   키워드×일 쇼핑 클릭 트렌드
│   ├── absolute.csv                                          #   역산 절대값
│   ├── gender_m.csv                                          #   남성 클릭 트렌드
│   ├── gender_f.csv                                          #   여성 클릭 트렌드
│   ├── age_10.csv                                            #   10대 클릭 트렌드
│   ├── age_20.csv                                            #   20대 클릭 트렌드
│   ├── age_30.csv                                            #   30대 클릭 트렌드
│   ├── age_40.csv                                            #   40대 클릭 트렌드
│   ├── age_50.csv                                            #   50대 클릭 트렌드
│   ├── age_60.csv                                            #   60대 이상 클릭 트렌드
│   ├── gender_m_pct.csv                                      #   남성 비율(%)
│   ├── gender_f_pct.csv                                      #   여성 비율(%)
│   ├── age_10_pct.csv                                        #   10대 비율(%)
│   ├── age_20_pct.csv                                        #   20대 비율(%)
│   ├── age_30_pct.csv                                        #   30대 비율(%)
│   ├── age_40_pct.csv                                        #   40대 비율(%)
│   ├── age_50_pct.csv                                        #   50대 비율(%)
│   └── age_60_pct.csv                                        #   60대 이상 비율(%)
└── search/                                                   # 통합 검색 트렌드 산출물
    ├── trend.csv                                             #   키워드×일 통합 검색 트렌드
    ├── absolute.csv                                          #   역산 절대값
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

### 트렌드 CSV 구조

행이 키워드, 열이 날짜(2022-01-01 ~ 2026-02-28)인 피벗 형태
키워드 존재 기간 외의 날짜는 NaN

**쇼핑 클릭 / 검색 (키워드 단독 조회)**

- `shopping_click/trend.csv`: 네이버 쇼핑 내 상품 클릭량 기반
- `search/trend.csv`: 네이버 통합 검색량 기반
- `*/absolute.csv`: 역산으로 추정한 절대값

값은 `/v1/datalab/shopping/category/keywords` (또는 `/v1/datalab/search`)가 반환하는 상대 비율
키워드 1개를 단독 조회하므로, 해당 키워드의 기간 내 최대 클릭(검색)일 = 100

**성별 / 연령별 (키워드별 그룹 분해 조회)**

- `shopping_click/gender_{m,f}.csv`: `/v1/datalab/shopping/category/keyword/gender`로 조회한 성별 쇼핑 클릭 트렌드
- `shopping_click/age_{10~60}.csv`: `/v1/datalab/shopping/category/keyword/age`로 조회한 연령별 쇼핑 클릭 트렌드

키워드 1개를 조회하면 응답에 전체 그룹(성별 2개 / 연령 6개)의 일별 ratio가 함께 담겨 옴
ratio는 해당 키워드의 전체 그룹 × 전체 기간을 통틀어 최대값 = 100으로 정규화
같은 키워드 내에서 그룹 간 ratio를 직접 비교 가능

예시 — "초코파이" 2024-06-01:
- 성별: m=17.32, f=11.59 → 남성 비율 = 17.32 / (17.32 + 11.59) = 59.9%
- 연령: 10대=1.01, 20대=6.54, 30대=5.37, 40대=12.58, 50대=10.91, 60대=4.19 → 40대 비율 = 12.58 / 40.60 = 31.0%

**성별 / 연령별 비율 (그룹 내 비율 파생)**

- `shopping_click/gender_{m,f}_pct.csv`: 성별 ratio에서 산출한 일별 남/여 비율(%)
- `shopping_click/age_{10~60}_pct.csv`: 연령별 ratio에서 산출한 일별 연령대 비율(%)

같은 키워드·같은 날짜에서 그룹별 ratio를 합산한 뒤 각 그룹이 차지하는 비율을 산출
전체 그룹이 NaN인 날짜는 NaN 유지, 일부 그룹만 NaN이면 0 취급 후 산출

예시 — "초코파이" 2024-06-01:
- gender_m_pct = 17.32 / (17.32 + 11.59) × 100 = 59.91%
- gender_f_pct = 11.59 / (17.32 + 11.59) × 100 = 40.09%
- age_40_pct = 12.58 / (1.01 + 6.54 + 5.37 + 12.58 + 10.91 + 4.19) × 100 = 31.0%

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
