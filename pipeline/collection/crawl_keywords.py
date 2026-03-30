# 네이버 데이터랩 쇼핑인사이트 키워드 크롤링
# 내부 API를 직접 호출하여 윈도우별 Top 500 키워드 수집
# 이미 저장된 윈도우는 스킵 (재실행 시 이어서 크롤링)

import json
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from dateutil.relativedelta import relativedelta
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
with open(PROJECT_ROOT / "configs" / "collection.yaml", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)["crawl"]

CID = _cfg["cid_2nd"]
CAT_1ST = _cfg["cat_1st_name"]
CAT_2ND = _cfg["cat_2nd_name"]
START = date.fromisoformat(_cfg["start_date"])
END = date.fromisoformat(_cfg["end_date"])
WINDOW_MONTHS = _cfg["window_months"]
PER_PAGE = 20
TOTAL_PAGES = _cfg["pages"]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "keyword_pool" / "windows"
API_URL = "https://datalab.naver.com/shoppingInsight/getCategoryKeywordRank.naver"
PAGE_URL = "https://datalab.naver.com/shoppingInsight/sCategory.naver"
EXPECTED_COLS = {"rank", "keyword", "window", "window_start", "window_end"}

JS_FETCH = """async ([url, body]) => {
    const r = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/x-www-form-urlencoded"},
        body: body
    });
    return await r.text();
}"""


def generate_windows():
    windows = []
    cursor = START
    while cursor < END:
        w_end = cursor + relativedelta(months=WINDOW_MONTHS, days=-1)
        if w_end >= END:
            w_end = END - relativedelta(days=1)
        windows.append((cursor, w_end))
        cursor += relativedelta(months=WINDOW_MONTHS)
    return windows


def csv_name(w_start, w_end):
    label = w_start.strftime("%Y-%m")
    return f"kw_{CAT_1ST}_{CAT_2ND}_{label}_{w_start:%Y%m%d}_{w_end:%Y%m%d}.csv"


# 브라우저 세션 내에서 API를 직접 호출하여 키워드 수집
def fetch_keywords(page, w_start, w_end):
    keywords = []
    for pg in range(1, TOTAL_PAGES + 1):
        body = f"cid={CID}&timeUnit=date&startDate={w_start}&endDate={w_end}&page={pg}&count={PER_PAGE}"
        raw = page.evaluate(JS_FETCH, [API_URL, body])
        if not raw:
            break
        data = json.loads(raw)
        ranks = data.get("ranks", [])
        if not ranks:
            break
        for r in ranks:
            keywords.append({"rank": r["rank"], "keyword": r["keyword"]})
        time.sleep(0.5)
    return keywords


# 윈도우별 CSV 검증 (컬럼 누락·빈 파일 제외)
def validate_csvs(csv_files):
    valid = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig", nrows=1)
        except Exception:
            continue
        if EXPECTED_COLS - set(df.columns):
            continue
        if len(pd.read_csv(f, encoding="utf-8-sig")) == 0:
            continue
        valid.append(f)
    return valid


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    windows = generate_windows()
    existing = {f.name for f in OUTPUT_DIR.glob("kw_*.csv")}
    todo = [(s, e) for s, e in windows if csv_name(s, e) not in existing]

    if todo:
        print(f"crawl_keywords: {len(todo)}/{len(windows)} windows to crawl")
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=False)
            ctx = browser.new_context()

            for i, (w_start, w_end) in enumerate(todo, 1):
                label = w_start.strftime("%Y-%m")

                # 윈도우마다 새 페이지로 세션 리셋 (레이트 리밋 우회)
                page = ctx.new_page()
                page.goto(PAGE_URL, wait_until="networkidle")
                keywords = fetch_keywords(page, w_start, w_end)
                page.close()

                df = pd.DataFrame(keywords)
                df["window"] = label
                df["window_start"] = w_start.isoformat()
                df["window_end"] = w_end.isoformat()
                df.to_csv(OUTPUT_DIR / csv_name(w_start, w_end), index=False, encoding="utf-8-sig")
                print(f"[{i}/{len(todo)}] {label}: {len(keywords)} keywords")

                time.sleep(3)

            browser.close()

    # 윈도우별 CSV 검증 + 통합
    csv_files = sorted(OUTPUT_DIR.glob("kw_*.csv"))
    valid = validate_csvs(csv_files)
    if not valid:
        return

    frames = [pd.read_csv(f, encoding="utf-8-sig") for f in valid]
    df = pd.concat(frames, ignore_index=True)

    all_path = OUTPUT_DIR.parent / f"keywords_all_{CAT_1ST}_{CAT_2ND}.csv"
    df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print(f"merged: {all_path.name} ({len(df)} rows, {df['keyword'].nunique()} unique)")


if __name__ == "__main__":
    main()
