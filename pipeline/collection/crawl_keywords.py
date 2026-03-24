# 네이버 데이터랩 쇼핑인사이트 키워드 크롤링
# 1개월 윈도우별로 Top 500 수집, 윈도우마다 개별 CSV 저장
# 이미 저장된 윈도우는 스킵 (재실행 시 이어서 크롤링)

import time
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from dateutil.relativedelta import relativedelta
from playwright.sync_api import Page, sync_playwright

# 프로젝트 루트 및 설정 로딩
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
with open(PROJECT_ROOT / "configs" / "collection.yaml", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)["crawl"]

CID_1ST       = _cfg["cid_1st"]
CID_2ND       = _cfg["cid_2nd"]
CAT_1ST_NAME  = _cfg["cat_1st_name"]
CAT_2ND_NAME  = _cfg["cat_2nd_name"]
START_DATE    = date.fromisoformat(_cfg["start_date"])
END_DATE      = date.fromisoformat(_cfg["end_date"])
WINDOW_MONTHS = _cfg["window_months"]
PAGES         = _cfg["pages"]
OUTPUT_DIR    = PROJECT_ROOT / "data" / "raw" / "keyword_windows"
URL           = "https://datalab.naver.com/shoppingInsight/sCategory.naver"


def csv_name(w_start: date, w_end: date) -> str:
    # kw_식품_과자베이커리_2022-01_20220101_20220131.csv
    label = w_start.strftime("%Y-%m")
    s = w_start.strftime("%Y%m%d")
    e = w_end.strftime("%Y%m%d")
    return f"kw_{CAT_1ST_NAME}_{CAT_2ND_NAME}_{label}_{s}_{e}.csv"


def generate_windows(start: date, end: date, months: int) -> list[tuple[date, date]]:
    windows = []
    cursor = start
    while cursor < end:
        window_end = cursor + relativedelta(months=months, days=-1)
        if window_end >= end:
            window_end = end - relativedelta(days=1)
        windows.append((cursor, window_end))
        cursor += relativedelta(months=months)
    return windows


def select_dropdown(select_div, *, data_cid: str | None = None, text: str | None = None) -> None:
    select_div.query_selector("span.select_btn").click()
    time.sleep(0.3)
    if data_cid:
        option = select_div.query_selector(f'a.option[data-cid="{data_cid}"]')
    else:
        options = select_div.query_selector_all("a.option")
        option = next((o for o in options if o.inner_text().strip() == text), None)
    if option is None:
        raise ValueError(f"옵션 못 찾음: data_cid={data_cid}, text={text}")
    option.click()
    time.sleep(0.3)


def _set_ymd(container, d: date) -> None:
    selects = container.query_selector_all("div.select")
    select_dropdown(selects[0], text=str(d.year))
    time.sleep(0.3)
    select_dropdown(selects[1], text=f"{d.month:02d}")
    time.sleep(0.3)
    select_dropdown(selects[2], text=f"{d.day:02d}")
    time.sleep(0.2)


def set_date_range(page: Page, start: date, end: date) -> None:
    inquiry = page.query_selector("div.section.insite_inquiry")
    inquiry.query_selector('label[data-index="3"]').click()
    time.sleep(0.5)
    target = inquiry.query_selector("div.set_period_target")
    spans = target.query_selector_all(":scope > span")
    _set_ymd(spans[0], start)
    time.sleep(0.3)
    _set_ymd(spans[2], end)
    time.sleep(0.3)


def crawl_window(page: Page, w_start: date, w_end: date) -> list[dict]:
    page.goto(URL, wait_until="networkidle")
    time.sleep(1)

    # 카테고리
    inquiry = page.query_selector("div.section.insite_inquiry")
    cat_area = inquiry.query_selector("div.set_period.category")
    selects = cat_area.query_selector_all("div.select")
    select_dropdown(selects[0], data_cid=CID_1ST)
    time.sleep(1)
    selects = cat_area.query_selector_all("div.select")
    select_dropdown(selects[1], data_cid=CID_2ND)
    time.sleep(0.5)

    # 날짜 + 조회
    set_date_range(page, w_start, w_end)
    inquiry.query_selector("a.btn_submit").click()
    page.wait_for_selector("div.rank_top1000 ul.rank_top1000_list li", timeout=15_000)
    time.sleep(1)

    # 페이지 순회
    keywords = []
    for page_num in range(1, PAGES + 1):
        items = page.query_selector_all("div.rank_top1000 ul.rank_top1000_list li")
        if not items:
            break
        for item in items:
            rank_el = item.query_selector("span.rank_top1000_num")
            link_el = item.query_selector("a.link_text")
            if rank_el and link_el:
                rank = int(rank_el.inner_text().strip())
                keyword = link_el.inner_text().strip().split("\n", 1)[-1].strip()
                keywords.append({"rank": rank, "keyword": keyword})
        if page_num < PAGES:
            next_btn = page.query_selector("a.btn_page_next")
            if not next_btn:
                break
            next_btn.click()
            time.sleep(0.8)

    return keywords


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = generate_windows(START_DATE, END_DATE, WINDOW_MONTHS)

    # 이미 저장된 윈도우 스킵
    existing = {f.name for f in OUTPUT_DIR.glob("kw_*.csv")}
    todo = [(s, e) for s, e in windows if csv_name(s, e) not in existing]

    print(f"전체 {len(windows)}개 윈도우, 완료 {len(windows) - len(todo)}개, 남은 {len(todo)}개\n")
    if not todo:
        print("모든 윈도우 크롤링 완료")
        return

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        page = browser.new_page()

        for i, (w_start, w_end) in enumerate(todo, 1):
            label = w_start.strftime("%Y-%m")
            print(f"[{i}/{len(todo)}] {label} ({w_start} ~ {w_end})")

            try:
                keywords = crawl_window(page, w_start, w_end)
            except Exception as e:
                print(f"  실패: {e}")
                continue

            # 윈도우별 CSV 저장
            df = pd.DataFrame(keywords)
            df["window"] = label
            df["window_start"] = w_start.isoformat()
            df["window_end"] = w_end.isoformat()
            path = OUTPUT_DIR / csv_name(w_start, w_end)
            df.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"  {len(keywords)}개 → {path.name}")

            if i < len(todo):
                time.sleep(2)

        browser.close()

    print("\n크롤링 완료")


if __name__ == "__main__":
    main()
