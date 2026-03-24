# 키워드 수집 파이프라인

## 개요

네이버 데이터랩 쇼핑인사이트에서 식품 > 과자/베이커리 키워드를 수집
전체 기간을 한번에 조회하면 스테디셀러가 상위를 독점하므로, 1개월 단위 윈도우로 쪼개서 각각 Top 500을 수집
특정 시기에만 반짝 등장한 바이럴 키워드도 포착 가능

네이버 데이터랩 내부 API(`getCategoryKeywordRank.naver`)를 직접 호출하는 방식
Playwright는 세션 쿠키 획득 용도로만 사용, UI 조작 없음

## 파일 구조

```
pipeline/collection/
├── crawl_keywords.py       # 1단계: API 호출 → 윈도우별 CSV 저장
├── aggregate_keywords.py   # 2단계: CSV 검증 → 통합 → 그룹별 선정
├── run_all.sh              # 배치 실행 (1단계 → 2단계)
└── README.md
```

## 실행 방법

```bash
# 전체 실행
./run_all.sh

# 또는 단계별
python pipeline/collection/crawl_keywords.py
python pipeline/collection/aggregate_keywords.py
```

## 산출물

```
data/raw/
├── keyword_windows/                                          # 윈도우별 중간산출물
│   ├── kw_식품_과자베이커리_2022-01_20220101_20220131.csv
│   ├── ...
│   └── kw_식품_과자베이커리_2026-02_20260201_20260228.csv     # 총 50개
├── keywords_all_식품_과자베이커리.csv                          # 전체 통합 (25,000행)
└── keywords_selected_500_seed{시드}.csv                       # 최종 선정 500개
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

aggregate:
  final_n: 500
  min_appearance: 4
  seed: null                 # null이면 매 실행마다 랜덤 시드
  strata:
    Q1_rare: 125
    Q2_low: 125
    Q3_mid: 125
    Q4_freq: 125
```

## 재실행 안전성

`crawl_keywords.py`는 이미 저장된 CSV를 확인하고 스킵
중단 후 재실행하면 남은 윈도우만 이어서 크롤링
