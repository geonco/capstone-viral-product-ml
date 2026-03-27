# 썸트렌드 데이터 수집 및 전처리 파이프라인

Claude(claude-sonnet-4-6) 활용 데이터 수집·병합·피처 생성 작업 기록

---

## 1. 폴더 구조

```
capstone-ml-study-main/           ← 로컬 작업 디렉토리 (git 미관리)
├── XLSX/                          # 썸트렌드 원본 xlsx 임시 보관
├── Sometrend2/                    # 키워드별 하위 폴더
│   └── 키워드명/
│       ├── 썸트렌드_키워드명_언급량_220301-230228.xlsx
│       ├── 썸트렌드_키워드명_언급량_230301-240229.xlsx
│       ├── 썸트렌드_키워드명_언급량_240301-250228.xlsx
│       ├── 썸트렌드_키워드명_언급량_250301-260228.xlsx
│       └── 키워드명_언급량_merged.csv
└── sometrend_merged/
    ├── sometrend_mention_long.csv      # 전체 키워드 통합 Long 포맷 (원본)
    └── sometrend_mention_features.csv  # 피처 생성 결과물

capstone-viral-product-ml/            ← 팀 레포
├── pipeline/
│   ├── preprocessing/
│   │   └── Sometrend_scripts/
│   │       ├── move_xlsx_to_sometrend2.py
│   │       ├── (클릭해서실행)run_move_xlsx.bat
│   │       ├── (클릭해서실행)check_missing_ranges.bat
│   │       ├── filter_date_range.py
│   │       ├── (클릭해서실행)filter_date_range.bat
│   │       ├── merge_sometrend.py
│   │       ├── xlsx_to_csv.py
│   │       └── *.bat
│   └── features/
│       ├── build_mention_features.py
│       └── (클릭해서실행)build_mention_features.bat
├── data/
│   ├── raw/                           # Sometrend2/, keywords_selected_500_*.csv
│   └── interim/                       # sometrend_mention_long.csv
└── docs/
    └── sometrend_data_pipeline.md     # 이 문서
```

---

## 2. 썸트렌드 파일 다운로드 규칙

- 파일명 형식: `썸트렌드_키워드명_언급량_YYMMDD-YYMMDD.xlsx`
- 기대 구간 4개 (키워드당):

  | 구간 | 기간 |
  |------|------|
  | 220301-230228 | 2022.03.01 ~ 2023.02.28 |
  | 230301-240229 | 2023.03.01 ~ 2024.02.29 |
  | 240301-250228 | 2024.03.01 ~ 2025.02.28 |
  | 250301-260228 | 2025.03.01 ~ 2026.02.28 |

- 데이터 자체가 없는 키워드 → 해당 구간 파일 다운로드 불가 → **0-fill 처리**

---

## 3. XLSX 분류 방식

1. XLSX 폴더에 원본 xlsx 파일 넣기
2. `(클릭해서실행)run_move_xlsx.bat` 더블클릭
3. 파일명에서 키워드 추출: `썸트렌드_(.+?)_언급량_` 정규식
4. Sometrend2 하위 폴더 중 이름이 완전히 일치하는 폴더에 파일 이동
   - 유사 키워드 혼용 방지를 위해 완전 일치만 허용
5. 폴더를 찾지 못한 파일은 "매칭 실패"로 출력

---

## 4. 키워드별 병합 방식

출력: `Sometrend2/키워드명/키워드명_언급량_merged.csv`

### xlsx 읽기
- `sheet_name=0`, `header=13` (썸트렌드 고정 포맷)
- 첫 번째 컬럼 = 날짜 (`2022.03.01` 형식 → pandas datetime 파싱)

### 병합 처리
1. 키워드 폴더 내 모든 xlsx `concat`
2. `drop_duplicates(subset=[날짜])` → 날짜 기준 중복 제거 (첫 번째 값 유지)
3. `sort_values(날짜)` → 날짜 오름차순 정렬

### 0-fill
- 전체 기간: **2022-01-01 ~ 2026-03-23**
- `reindex` + `fillna(0)` → 데이터 없는 날짜는 0으로 채움
- 결과: 항상 **1,543행** (데이터 유무 관계없이 동일)

### 저장
- 파일명: `키워드명_언급량_merged.csv`
- 인코딩: `utf-8-sig`
- 날짜 형식: `%Y.%m.%d`
- 컬럼: `날짜`, `합계`, `커뮤니티`, `인스타그램`, `블로그`, `뉴스`

---

## 5. 전체 통합 Long 포맷 생성

출력: `sometrend_merged/sometrend_mention_long.csv`

### 형태
- Long 포맷: (keyword, date) 쌍이 각 행
- 행 수: 487개 키워드 × 1,543일 = **749,617행**
- 컬럼: `keyword`, `date`, `합계`, `커뮤니티`, `인스타그램`, `블로그`, `뉴스`

---

## 6. 수집 기간 필터링

스크립트: `pipeline/preprocessing/Sometrend_scripts/filter_date_range.py`
실행: `(클릭해서실행)filter_date_range.bat` 더블클릭

- 수집 기간 외 행 제거: **2022-03-01 ~ 2026-02-28**
- 원본 파일 덮어쓰기
- 0-fill 구간(2022-01-01 ~ 2022-02-28, 2026-03-01 ~) 제거

---

## 7. 피처 생성

스크립트: `pipeline/features/build_mention_features.py`
실행: `(클릭해서실행)build_mention_features.bat` 더블클릭
출력: `sometrend_merged/sometrend_mention_features.csv` (신규 생성, 원본 유지)

### 피처 목록 (합계 컬럼 기준)

| Feature | 계산 방법 | 의미 |
|---------|-----------|------|
| `mention_ma_3d` | 최근 3일 평균 | 초단기 buzz |
| `mention_ma_7d` | 최근 7일 평균 | 단기 buzz 강도 |
| `mention_growth_3d` | 최근 3일 증가율 | hype 급증 |
| `mention_growth_7d` | 최근 7일 증가율 | buzz 상승 속도 |
| `mention_growth_14d` | 최근 14일 증가율 | 중기 buzz trend |
| `mention_acceleration` | growth_7d − growth_14d | hype 가속 |
| `mention_std_7d` | 최근 7일 std | buzz 변동성 |
| `mention_pos_14d` | 현재 / 최근 14일 max | buzz peak 위치 |
| `days_since_mention_max_14d` | peak 이후 경과일 | buzz timing |

---

## 8. 스크립트 목록

| 파일 | 위치 | 용도 | 실행 방법 |
|------|------|------|-----------|
| `move_xlsx_to_sometrend2.py` | preprocessing/Sometrend_scripts/ | XLSX 분류 + 빠진 구간 보고 + 키워드별 병합 | bat 더블클릭 |
| `filter_date_range.py` | preprocessing/Sometrend_scripts/ | 수집 기간 외 행 제거 | bat 더블클릭 |
| `merge_sometrend.py` | preprocessing/Sometrend_scripts/ | 구버전 병합 스크립트 (참고용) | — |
| `xlsx_to_csv.py` | preprocessing/Sometrend_scripts/ | xlsx → csv 단순 변환 유틸리티 | — |
| `build_mention_features.py` | features/ | 언급량 피처 9종 생성 | bat 더블클릭 |

---

## 9. 데이터 없음 처리

- 폴더명에 `(데이터X)`, `(데이터 미존재)`, `(데이터없음)` 등 표기
- 병합 집계 및 통합 파일에서 제외

---

## 10. 작업 현황 (2026-03-27 기준)

| 항목 | 수치 |
|------|------|
| 전체 키워드 폴더 | 507개 |
| 데이터없음 제외 실질 대상 | 488개 |
| 키워드별 병합 완료 | 487개 |
| 통합 Long 파일 수록 키워드 | 487개 |
| 날짜 필터링 | 완료 (2022-03-01 ~ 2026-02-28) |
| 피처 생성 | 완료 (mention 9종) |
