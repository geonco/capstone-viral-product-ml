# VIRAL — 식품 키워드 바이럴 예측 (프론트엔드 프로토타입)

Next.js 14 기반 대시보드. 마피아넷 스타일 + 모던 다크 톤. 백엔드 없이 mock JSON으로 동작

## 페이지

- `/` 메인 — 검색바 + 오늘의 랭킹 + 4개 클러스터 (급상승/지속/피크임박/하락주의)
- `/keyword/[name]` 상세 — KPI 4개 + 4채널 시계열 + 레이더 + 인구통계 + SHAP + 연관 키워드
- `/compare` 비교 — 최대 5개 키워드 시계열·KPI 동시 비교
- `/explore` 탐색 — 성장×지속성 사분면 산점도
- `/category/[name]` 카테고리별 랭킹

## 실행

```bash
# 1. mock 데이터 생성 (한 번만)
cd ..
python scripts/build_frontend_mock.py

# 2. 의존성 설치 + 개발 서버
cd frontend
npm install
npm run dev
```

http://localhost:3000

## 스택

- Next.js 14 App Router + TypeScript
- Tailwind CSS (다크 테마 커스텀 색상)
- Recharts (시계열·산점도·레이더·도넛·바 차트)
- lucide-react (아이콘)

## 데이터 흐름

`scripts/build_frontend_mock.py`가 `data/raw/`에서 키워드 20개를 추출해 `frontend/public/mock/` 아래 JSON으로 떨어뜨림. `lib/data.ts`가 서버 컴포넌트에서 fs로 읽음. 실서비스 시 fetch로 갈아끼우기만 하면 됨

## 라벨 매핑 (현재 모델 → 화면)

- `growth_10d` → 14일 성장 전망 (메인 KPI)
- `sustainability_10d` → 지속성
- `fw_peak_softpos_10d` → 피크 시점 (LSTM)
- `buzz_composite_10d` → 현재 버즈 강도
- `crash_10d` → 하락 위험
- `spike_10d` → 급등 패턴
