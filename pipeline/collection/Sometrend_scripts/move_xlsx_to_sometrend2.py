"""
XLSX 폴더의 파일을 키워드 기준으로 Sometrend2 하위 폴더에 이동 + 자동 병합.

사용법:
    python move_xlsx_to_sometrend2.py [xlsx_dir] [sometrend2_dir]

기본값:
    xlsx_dir       = XLSX
    sometrend2_dir = Sometrend2
"""

import re
import shutil
import sys
from pathlib import Path

import pandas as pd

# ── 0-fill 기간 ────────────────────────────────────────────────────────────────
FILL_START = "2022-01-01"
FILL_END   = "2026-03-23"

# 기대하는 4개 구간 (파일명 suffix 기준)
EXPECTED_RANGES = [
    "220301-230228",
    "230301-240229",
    "240301-250228",
    "250301-260228",
]


# ── 빠진 구간 체크 ─────────────────────────────────────────────────────────────
def check_missing_ranges(folder: Path, keyword: str) -> list[str]:
    """키워드 폴더 안의 xlsx 파일 중 없는 기대 구간 목록을 반환."""
    existing = {f.name for f in folder.glob("*.xlsx")}
    missing = []
    for rng in EXPECTED_RANGES:
        expected_name = f"썸트렌드_{keyword}_언급량_{rng}.xlsx"
        if expected_name not in existing:
            missing.append(rng)
    return missing


# ── 병합 ───────────────────────────────────────────────────────────────────────
def merge_keyword(folder: Path, keyword: str):
    files = sorted(folder.glob("*.xlsx"))
    if not files:
        print(f"  [!] xlsx 파일 없음, 건너뜀")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_excel(f, sheet_name=0, header=13)
        except Exception as e:
            print(f"  [!] 읽기 실패 {f.name}: {e}")
            continue
        df = df.dropna(how="all")
        col0 = df.columns[0]
        df[col0] = pd.to_datetime(
            df[col0].astype(str).str.replace(".", "-", regex=False),
            errors="coerce",
        )
        df = df.dropna(subset=[col0])
        dfs.append(df)

    if not dfs:
        print(f"  [!] 유효한 데이터 없음, 건너뜀")
        return

    merged = pd.concat(dfs, ignore_index=True)
    col0 = merged.columns[0]
    merged = (
        merged.drop_duplicates(subset=[col0])
        .sort_values(col0)
        .reset_index(drop=True)
    )

    full_range = pd.date_range(FILL_START, FILL_END)
    merged = (
        merged.set_index(col0)
        .reindex(full_range)
        .fillna(0)
        .reset_index()
    )
    merged.rename(columns={"index": col0}, inplace=True)
    merged[col0] = merged[col0].dt.strftime("%Y.%m.%d")

    out = folder / f"{keyword}_언급량_merged.csv"
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"  → 병합 완료: {out.name}  ({len(merged)}행, {len(files)}개 파일)")


# ── 이동 + 병합 ────────────────────────────────────────────────────────────────
def move_and_merge(xlsx_dir: Path, sometrend2_dir: Path):
    files = sorted(xlsx_dir.glob("*.xlsx"))
    if not files:
        print(f"[!] {xlsx_dir} 안에 xlsx 파일이 없습니다.")
        return

    moved_keywords: set[str] = set()
    skipped: list[str] = []
    not_found: list[str] = []

    # 1단계: 이동
    for f in files:
        m = re.match(r"썸트렌드_(.+?)_언급량_", f.name)
        if not m:
            skipped.append(f.name)
            continue
        keyword = m.group(1)
        target_dir = sometrend2_dir / keyword
        if target_dir.is_dir():
            dest = target_dir / f.name
            if dest.exists():
                skipped.append(f"{f.name} (이미 존재)")
            else:
                shutil.move(str(f), str(dest))
                moved_keywords.add(keyword)
                print(f"[이동] {keyword}  ←  {f.name}")
        else:
            not_found.append(f"{keyword}: {f.name}")

    if skipped:
        print(f"\n건너뜀: {len(skipped)}개")
        for s in skipped:
            print(f"  {s}")

    if not_found:
        print(f"\n매칭 실패 (폴더 없음): {len(not_found)}개")
        for n in not_found:
            print(f"  {n}")

    if not moved_keywords:
        print("\n이동된 파일이 없어 병합을 건너뜁니다.")
        return

    # 2단계: 빠진 구간 보고 + 병합
    print(f"\n{'='*60}")
    print(f"[병합 시작]  {len(moved_keywords)}개 키워드")
    print(f"{'='*60}")

    for keyword in sorted(moved_keywords):
        folder = sometrend2_dir / keyword
        missing = check_missing_ranges(folder, keyword)
        if missing:
            print(f"\n[{keyword}]  빠진 구간: {', '.join(missing)}")
        else:
            print(f"\n[{keyword}]  구간 완전")
        merge_keyword(folder, keyword)


# ── 전체 폴더 빠진 구간 스캔 ──────────────────────────────────────────────────
def check_all(sometrend2_dir: Path):
    """Sometrend2 하위 폴더 전체를 스캔해서 빠진 구간을 키워드별로 출력."""
    folders = sorted(p for p in sometrend2_dir.iterdir() if p.is_dir())
    if not folders:
        print("[!] Sometrend2 하위 폴더가 없습니다.")
        return

    complete = []
    incomplete = []  # (keyword, missing_list)

    for folder in folders:
        keyword = folder.name
        missing = check_missing_ranges(folder, keyword)
        if missing:
            incomplete.append((keyword, missing))
        else:
            complete.append(keyword)

    print(f"{'='*60}")
    print(f"Sometrend2 전체 구간 점검  ({len(folders)}개 키워드)")
    print(f"{'='*60}")

    if incomplete:
        print(f"\n[빠진 구간 있음]  {len(incomplete)}개 키워드")
        for keyword, missing in incomplete:
            print(f"  {keyword:30s}  빠진 구간: {', '.join(missing)}")
    else:
        print("\n빠진 구간 없음.")

    print(f"\n[구간 완전]  {len(complete)}개 키워드")
    for kw in complete:
        print(f"  {kw}")


# ── 진입점 ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    base = Path(__file__).parent

    parser = argparse.ArgumentParser(description="XLSX 이동 + 병합 스크립트")
    parser.add_argument("xlsx_dir",       nargs="?", default=str(base / "XLSX"),
                        help="XLSX 원본 폴더 (기본: XLSX)")
    parser.add_argument("sometrend2_dir", nargs="?", default=str(base / "Sometrend2"),
                        help="Sometrend2 폴더 (기본: Sometrend2)")
    parser.add_argument("--check", action="store_true",
                        help="Sometrend2 전체 폴더 빠진 구간 점검만 수행 (이동/병합 없음)")
    args = parser.parse_args()

    xlsx_dir       = Path(args.xlsx_dir)
    sometrend2_dir = Path(args.sometrend2_dir)

    if not sometrend2_dir.is_dir():
        print(f"[ERR] Sometrend2 폴더를 찾을 수 없습니다: {sometrend2_dir}")
        sys.exit(1)

    if args.check:
        print(f"Sometrend2 폴더: {sometrend2_dir}\n")
        check_all(sometrend2_dir)
    else:
        if not xlsx_dir.is_dir():
            print(f"[ERR] XLSX 폴더를 찾을 수 없습니다: {xlsx_dir}")
            sys.exit(1)
        print(f"XLSX 폴더     : {xlsx_dir}")
        print(f"Sometrend2 폴더: {sometrend2_dir}\n")
        move_and_merge(xlsx_dir, sometrend2_dir)
