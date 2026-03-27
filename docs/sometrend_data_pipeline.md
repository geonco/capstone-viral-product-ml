# 썸트렌드 데이터 수집 및 병합 작업방식

Claude(claude-sonnet-4-6)를 활용하여 진행한 데이터 수집 및 병합 작업 기록.

---

## 1. 폴더 구조

```
capstone-ml-study-main/
├── XLSX/                          # 썸트렌드에서 다운로드한 원본 xlsx 임시 보관
├── Sometrend2/                    # 키워드별 하위 폴더
│   ├── 키워드명/
│   │   ├── 썸트렌드_키워드명_언급량_220301-230228.xlsx
│   │   ├── 썸트렌드_키워드명_언급량_230301-240229.xlsx
│   │   ├── 썸트렌드_키워드명_언급량_240301-250228.xlsx
│   │   ├── 썸트렌드_키워드명_언급량_250301-260228.xlsx
│   │   └── 키워드명_언급량_merged.csv   ← 키워드별 병합 결과물
│   └── ...
├── sometrend_merged/
│   └── sometrend_mention_long.csv  ← 전체 키워드 통합 Long 포맷
└── scripts/                        # 작업 스크립트 모음
    ├── move_xlsx_to_sometrend2.py
    ├── (클릭해서실행)run_move_xlsx.bat
    ├── (클릭해서실행)check_missing_ranges.bat
    ├── merge_sometrend.py
    ├── run_merge_sometrend.bat
    ├── xlsx_to_csv.py
    └── run_xlsx_to_csv.bat
```

---

## 2. 썸트렌드 파일 다운로드 규칙

- **파일명 형식**: `썸트렌드_키워드명_언급량_YYMMDD-YYMMDD.xlsx`
- **기대 구간 4개** (키워드당):

  | 구간 | 기간 |
  |------|------|
  | 220301-230228 | 2022.03.01 ~ 2023.02.28 |
  | 230301-240229 | 2023.03.01 ~ 2024.02.29 |
  | 240301-250228 | 2024.03.01 ~ 2025.02.28 |
  | 250301-260228 | 2025.03.01 ~ 2026.02.28 |

- 데이터 자체가 없는 키워드는 해당 구간 파일을 다운로드할 수 없음 → **0-fill 처리**

---

## 3. 분류 방식

1. XLSX 폴더에 원본 xlsx 파일 넣기
2. `(클릭해서실행)run_move_xlsx.bat` 더블클릭
3. 파일명에서 키워드 추출: `썸트렌드_(.+?)_언급량_` 정규식
4. Sometrend2 하위 폴더 중 **이름이 완전히 일치하는 폴더**에 파일 이동
   - 유사 키워드(예: 만다라 vs 만만다라) 혼용 방지를 위해 완전 일치만 허용
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

### 저장
- 파일명: `키워드명_언급량_merged.csv`
- 인코딩: `utf-8-sig` (Excel 호환)
- 날짜 형식: `%Y.%m.%d` (점 구분)
- 컬럼: `날짜`, `합계`, `커뮤니티`, `인스타그램`, `블로그`, `뉴스`

---

## 5. 전체 통합 Long 포맷 생성 (→ sometrend_merged/sometrend_mention_long.csv)

키워드별 merged.csv를 하나로 합쳐 ML 파이프라인 입력용 통합 파일로 생성.

### 형태
- **Long 포맷**: (keyword, date) 쌍이 각 행
- 행 수: 487개 키워드 × 1,543일 = **749,617행**
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
result.to_csv('sometrend_merged/sometrend_mention_long.csv', index=False, encoding='utf-8-sig')
```

---

## 6. 스크립트 목록 (scripts/)

| 파일 | 용도 |
|------|------|
| `move_xlsx_to_sometrend2.py` | XLSX 분류 + 빠진 구간 보고 + 키워드별 병합 메인 스크립트 |
| `(클릭해서실행)run_move_xlsx.bat` | 위 스크립트 실행 (더블클릭) |
| `(클릭해서실행)check_missing_ranges.bat` | Sometrend2 전체 빠진 구간 점검만 수행 |
| `merge_sometrend.py` | 구버전 병합 스크립트 (참고용) |
| `run_merge_sometrend.bat` | 구버전 병합 bat |
| `xlsx_to_csv.py` | xlsx → csv 단순 변환 유틸리티 |
| `run_xlsx_to_csv.bat` | 위 스크립트 실행 bat |

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
