# 썸트렌드 데이터 수집 및 병합 작업방식

Claude(claude-sonnet-4-6)를 활용하여 진행한 데이터 수집 및 병합 작업 기록.

---

## 0. 데이터 수집 목적

```
네이버 데이터랩 API 에서 추출한 키워드 500개를 대상으로 https://some.co.kr/ 에서 수동으로 검색을 진행하여, 2022-03-01 부터 2026-02-28 까지의 데이터를 수집한다. 수집한 데이터는 그 키워드가 얼마나 언급되었는지를 나타내며, 커뮤니티, 인스타그램, 블로그, 뉴스 등으로 분류되어 있음. 언급량이 기재된 자료를 스크립트를 통해 정리하고, 이를 통해 시간에 따른 클릭 평균, 클릭 증가량, 표준 편차를 계산하여 모델의 학습 능력 및 예측 능력을 향상하고자 함.


## 1. 폴더 구조

수집 작업은 별도 로컬 환경에서 수행 후, 최종 산출물만 프로젝트에 반입.

```
(팀원 로컬 작업 환경 — 프로젝트 외부)
├── XLSX/                          # 썸트렌드에서 다운로드한 원본 xlsx 임시 보관
├── Sometrend2/                    # 키워드별 하위 폴더
│   ├── 키워드명/
│   │   ├── 썸트렌드_키워드명_언급량_220301-230228.xlsx
│   │   ├── 썸트렌드_키워드명_언급량_230301-240229.xlsx
│   │   ├── 썸트렌드_키워드명_언급량_240301-250228.xlsx
│   │   ├── 썸트렌드_키워드명_언급량_250301-260228.xlsx
│   │   └── 키워드명_언급량_merged.csv   ← 키워드별 병합 결과물
│   └── ...

(이 프로젝트)
capstone-viral-product-ml/
├── pipeline/collection/Sometrend_scripts/   # 수집·병합 스크립트
│   ├── move_xlsx_to_sometrend2.py
│   ├── merge_sometrend.py
│   └── run_merge_sometrend.bat
└── data/raw/
    └── sometrend_mention_long.csv           # 최종 산출물 (RAW DATA)
```

---

## 2. 썸트렌드 파일 다운로드 규칙
- 썸트렌드는 자체적으로 크롤링을 금지하고 있기 때문에 BeautifulSoap등의 라이브러리를 사용할 수 없으므로, 주어진 키워드마다, 1년 단위로 검색을 반복하여 2022-03-01 ~ 2026-02-28 까지의 데이터를 확보한다.
    그 후, 다음과 같은 규칙에 따라, 다운로드받은 xlsx 파일들을 키워드에 따라 분류한 후, 2022-03-31 ~ 2026-02-28 사이의 데이터가 한꺼번에 들어가있는 파일을 만든다.
- 파일명 형식: `썸트렌드_키워드명_언급량_YYMMDD-YYMMDD.xlsx`
- 기대 구간 4개 (키워드당):

  | 구간 | 기간 |
  |------|------|
  | 220301-230228 | 2022.03.01 ~ 2023.02.28 |
  | 230301-240229 | 2023.03.01 ~ 2024.02.29 |
  | 240301-250228 | 2024.03.01 ~ 2025.02.28 |
  | 250301-260228 | 2025.03.01 ~ 2026.02.28 |

- 데이터가 검색되지 않는 구간이 일부 존재할 경우 → 최근에 바이럴이 되어 이전 데이터가 없거나, 언급량이 떨어졌을 가능성이 있으므로 0으로 채운다.
- 모든 구간에서 데이터가 검색되지 않을 경우 -> 특정 Feature들이 모델의 학습에 왜곡을 일으킬 수 있으므로 빈 폴더로 방기.

---

## 3. 분류 방식

1. XLSX 폴더에 원본 xlsx 파일 넣기
2. `(클릭해서실행)run_move_xlsx.bat` 더블클릭
3. 파일명에서 키워드 추출: `썸트렌드_(.+?)_언급량_` 정규식
4. Sometrend2 하위 폴더 중 **이름이 완전히 일치하는 폴더**에 파일 이동
   - 유사 키워드(예: 초콜릿 vs 쵸콜릿) 혼용 방지를 위해 완전 일치만 허용
5. 폴더를 찾지 못한 파일은 "매칭 실패"로 출력

---

## 4. 키워드별 병합 방식 (→ Sometrend2/키워드명/키워드명_언급량_merged.csv)

### xlsx 읽기
- `sheet_name=0`, `header=13` (썸트렌드 고정 포맷)
- 첫 번째 컬럼 = 날짜 (`2022.03.01` 형식 → pandas datetime으로 파싱)

### 병합 처리
1. 키워드 폴더 내 모든 xlsx `concat`
2. `drop_duplicates(subset=[날짜])` → 날짜 기준 중복 제거 (첫 번째 값 유지)
3. `sort_values(날짜)` → 날짜 오름차순 정렬

### 0-fill
- 전체 기간: **2022-01-01 ~ 2026-03-23**
- `reindex` + `fillna(0)` → 데이터 없는 날짜는 0으로 채움
- 결과: 항상 **1,543행** (데이터 유무 관계없이 동일)
- 추후 스크립트를 통해 2022-01-01~2022-02-28, 2026-03-01~ 데이터는 폐기함.

### 저장
- 파일명: `키워드명_언급량_merged.csv`
- 인코딩: `utf-8-sig` (Excel 호환)
- 날짜 형식: `%Y.%m.%d` (점 구분)
- 컬럼: `날짜`, `합계`, `커뮤니티`, `인스타그램`, `블로그`, `뉴스`

---

## 5. 전체 통합 Long 포맷 생성 (→ data/raw/sometrend_mention_long.csv)

키워드별 merged.csv를 하나로 합쳐 ML 파이프라인 입력용 통합 파일로 생성.

### 형태
- **Long 포맷**: (keyword, date) 쌍이 각 행을 담당함.
- 행 수: 487개 키워드 × 1,543일 = **709,799행**
- 컬럼: `keyword`, `date`, `합계`, `커뮤니티`, `인스타그램`, `블로그`, `뉴스`

### 생성 방법
```python
import pandas as pd
from pathlib import Path

BASE = Path('Sometrend2')
dfs = []
for folder in sorted(BASE.iterdir()):
    if not folder.is_dir():
        continue
    csv_files = list(folder.glob('*_merged.csv'))
    if not csv_files:
        continue
    keyword = folder.name
    df = pd.read_csv(csv_files[0], encoding='utf-8-sig')
    col0 = df.columns[0]
    df = df.rename(columns={col0: 'date'})
    df.insert(0, 'keyword', keyword)
    dfs.append(df)

result = pd.concat(dfs, ignore_index=True)
result.to_csv('data/raw/sometrend_mention_long.csv', index=False, encoding='utf-8-sig')
```

---

## 6. 스크립트 목록 (pipeline/collection/Sometrend_scripts/)

| 파일 | 용도 |
|------|------|
| `move_xlsx_to_sometrend2.py` | XLSX 분류 + 빠진 구간 보고 + 키워드별 병합 메인 스크립트 |
| `merge_sometrend.py` | 전체 통합 long 포맷 생성 스크립트 |
| `run_merge_sometrend.bat` | merge_sometrend.py 실행 bat |

---

## 7. 데이터 없음 처리

- 폴더명에 `(데이터X)`, `(데이터 미존재)`, `(데이터없음)` 등 표기
- 병합 집계 및 통합 파일에서 제외
- 해당 키워드의 빠진 구간은 0-fill로 대체

---

## 8. 작업 현황 (2026-03-27 기준)

| 항목 | 수치 |
|------|------|
| 전체 키워드 폴더 | 507개 |
| 데이터없음 제외 실질 대상 | 488개 |
| 키워드별 병합 완료 | 487개 |
| 통합 Long 파일 수록 키워드 | 487개 |
| 미완료 | 1개 (갓은혜스콘 — 데이터 없음, 사실상 완료) |
