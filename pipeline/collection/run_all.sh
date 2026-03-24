#!/bin/bash
# 키워드 수집 배치 실행: 크롤링 → 검증/통합/선정
cd "$(dirname "$0")"
python crawl_keywords.py && python aggregate_keywords.py
