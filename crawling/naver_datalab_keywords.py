"""
네이버 데이터랩 쇼핑인사이트 - 식품 > 과자/베이커리 인기검색어 TOP 500 크롤러

Playwright를 사용하여 브라우저 자동화로 데이터를 수집합니다.
결과는 data_raw/naver_datalab_top500_keywords.csv에 저장됩니다.
"""

from playwright.sync_api import sync_playwright
import pandas as pd
import time
import os
from datetime import datetime

URL = "https://datalab.naver.com/shoppingInsight/sCategory.naver"
CID_FOOD = "50000006"          # 식품
CID_BAKERY = "50000149"        # 과자/베이커리
TOTAL_PAGES = 25               # 20개씩 25페이지 = 500개


def select_dropdown_option(select_div, data_cid=None, text=None):
    """커스텀 드롭다운에서 옵션을 선택합니다."""
    # 드롭다운 열기
    select_div.query_selector("span.select_btn").click()
    time.sleep(0.3)

    # 옵션 선택
    if data_cid:
        option = select_div.query_selector(f'a.option[data-cid="{data_cid}"]')
    elif text:
        options = select_div.query_selector_all("a.option")
        option = next((o for o in options if text in o.inner_text()), None)

    if option is None:
        raise ValueError(f"Option not found: data_cid={data_cid}, text={text}")
    option.click()
    time.sleep(0.5)


def extract_keywords(page):
    """현재 페이지에서 키워드 목록을 추출합니다."""
    keywords = []
    items = page.query_selector_all("div.rank_top1000 ul.rank_top1000_list li")
    for item in items:
        rank_el = item.query_selector("span.rank_top1000_num")
        link_el = item.query_selector("a.link_text")
        if rank_el and link_el:
            rank = int(rank_el.inner_text().strip())
            # inner_text에는 "1\n두쫀쿠" 형태이므로 rank 숫자를 제거
            full_text = link_el.inner_text().strip()
            keyword = full_text.split("\n", 1)[-1].strip()
            keywords.append({"rank": rank, "keyword": keyword})
    return keywords


def crawl_top500():
    """식품 > 과자/베이커리 인기검색어 TOP 500을 크롤링합니다."""
    all_keywords = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(URL, wait_until="networkidle")

        # 1. 카테고리 선택 - 첫 번째 조회 섹션
        inquiry = page.query_selector("div.section.insite_inquiry")
        cat_area = inquiry.query_selector("div.set_period.category")
        selects = cat_area.query_selector_all("div.select")

        # 1분류: 식품
        select_dropdown_option(selects[0], data_cid=CID_FOOD)
        print("1분류 '식품' 선택 완료")

        # 2분류 드롭다운이 갱신될 때까지 대기
        time.sleep(1)
        selects = cat_area.query_selector_all("div.select")

        # 2분류: 과자/베이커리
        select_dropdown_option(selects[1], data_cid=CID_BAKERY)
        print("2분류 '과자/베이커리' 선택 완료")

        # 2. 조회하기 클릭
        inquiry.query_selector("a.btn_submit").click()
        print("조회하기 클릭")

        # 결과 로딩 대기
        page.wait_for_selector("div.rank_top1000 ul.rank_top1000_list li", timeout=10000)
        time.sleep(1)

        # 3. 25페이지 순회하며 키워드 수집
        for page_num in range(1, TOTAL_PAGES + 1):
            keywords = extract_keywords(page)
            all_keywords.extend(keywords)
            print(f"  페이지 {page_num}/{TOTAL_PAGES}: {len(keywords)}개 수집 (누적 {len(all_keywords)}개)")

            if page_num < TOTAL_PAGES:
                next_btn = page.query_selector("a.btn_page_next")
                next_btn.click()
                time.sleep(0.5)

        browser.close()

    print(f"\n총 {len(all_keywords)}개 키워드 수집 완료")
    return all_keywords


def save_to_csv(keywords):
    """키워드 데이터를 CSV로 저장합니다."""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data_raw")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(keywords)
    df["crawled_date"] = datetime.now().strftime("%Y-%m-%d")

    output_path = os.path.join(output_dir, "naver_datalab_top500_keywords.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {output_path}")
    return output_path


if __name__ == "__main__":
    keywords = crawl_top500()
    save_to_csv(keywords)
