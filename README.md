# capstone-viral-product-ml

과자/베이커리 키워드의 바이럴 강도 예측 ML 프로젝트

## 디렉토리 구조

```
capstone-ml/
│
├── configs/                    # 설정값 중앙 관리 (YAML)
│
├── pipeline/                   # 모든 실행 코드
│   ├── collection/             #   Stage 1: 데이터 수집
│   ├── preprocessing/          #   Stage 2: 전처리
│   ├── features/               #   Stage 3: 피처·타겟 생성
│   ├── training/               #   Stage 4: 학습·평가
│   └── inference/              #   추론
│
├── lib/                        # 스테이지 간 공유 유틸리티
│
├── data/                       # 모든 데이터 (.gitignore)
│   ├── raw/                    #   수집 원본 (불변)
│   ├── interim/                #   전처리 중간 산출물
│   └── processed/              #   학습 준비 완료 데이터
│
├── outputs/                    # 실험 결과 (.gitignore)
│   ├── models/                 #   저장된 weight
│   ├── metrics/                #   평가 결과
│   └── figures/                #   SHAP plot, 차트
│
├── experiments/                # 탐색적 실험 (파이프라인 외)
├── notebooks/                  # EDA 전용 (프로덕션 코드 금지)
├── docs/                       # 설계 문서
├── tests/                      # 단위 테스트 (pytest)
│
├── requirements.txt
├── .env                        # API 키 (git 제외)
├── .gitignore
└── README.md
```

## 실행

```bash
# 가상환경 설정
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# 키워드 수집 파이프라인
python pipeline/collection/crawl_keywords.py
python pipeline/collection/select_keywords.py
python pipeline/collection/crawl_shopping_click_trend.py
python pipeline/collection/crawl_search_trend.py
```
