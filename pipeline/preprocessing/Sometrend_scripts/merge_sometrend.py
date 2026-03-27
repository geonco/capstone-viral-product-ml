"""
썸트렌드 파일 타입별 병합 스크립트 (xlsx / csv 모두 지원)
사용법: python merge_sometrend.py <키워드_폴더_경로>
예: python merge_sometrend.py Sometrend2/칸쵸
    python merge_sometrend.py Sometrend2/아이스크림콘
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

HEADER_ROW = 13  # 0-indexed (14번째 줄이 헤더)

# 0-fill 범위
FILL_START = "2022-01-01"
FILL_END   = "2026-03-23"


def apply_fill(merged: pd.DataFrame) -> pd.DataFrame:
    """날짜 0-fill 및 형식 통일 (%Y.%m.%d)"""
    col0 = merged.columns[0]
    merged = merged.copy()
    merged[col0] = pd.to_datetime(
        merged[col0].astype(str).str.replace(".", "-", regex=False), errors="coerce"
    )
    merged = merged.dropna(subset=[col0])
    full_range = pd.date_range(FILL_START, FILL_END)
    merged = merged.set_index(col0).reindex(full_range).fillna(0).reset_index()
    merged.rename(columns={"index": col0}, inplace=True)
    merged[col0] = merged[col0].dt.strftime("%Y.%m.%d")
    return merged

# 날짜 범위 패턴: _YYMMDD-YYMMDD 또는 _YYYYMMDD-YYYYMMDD
DATE_RANGE_RE = re.compile(r"_\d{6,8}-\d{6,8}")


def strip_date(stem: str) -> str:
    """파일명에서 날짜 범위 및 중복 다운로드 suffix( (1), (2) 등) 제거 → 그룹 키 반환"""
    result = DATE_RANGE_RE.sub("", stem)
    result = re.sub(r"\s*\(\d+\)\s*$", "", result)
    return result


def clean_key(group_key: str, keyword: str) -> str:
    """그룹 키에서 '썸트렌드_{keyword}' 또는 '썸트렌드_{keyword} ' 접두어 제거"""
    for prefix in (f"썸트렌드_{keyword}_", f"썸트렌드_{keyword} ", f"썸트렌드_{keyword}"):
        if group_key.startswith(prefix):
            return group_key[len(prefix):]
    return group_key


def merge_xlsx_folder(folder: Path):
    """xlsx 파일 기반 병합 (칸쵸 방식)"""
    files = sorted(folder.glob("*.xlsx"))
    if not files:
        return

    groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        key = strip_date(f.stem)
        groups[key].append(f)

    keyword = folder.name
    for group_key, group_files in sorted(groups.items()):
        group_files = sorted(group_files)
        print(f"\n  [{group_key}] {len(group_files)}개 파일")

        first_xl = pd.ExcelFile(group_files[0])
        sheet_names = first_xl.sheet_names

        for sheet in sheet_names:
            dfs = []
            for f in group_files:
                df = pd.read_excel(f, sheet_name=sheet, header=HEADER_ROW)
                df = df.dropna(how="all")
                dfs.append(df)

            merged = pd.concat(dfs, ignore_index=True)
            merged = merged.drop_duplicates(subset=[merged.columns[0]])
            merged = apply_fill(merged)

            safe_sheet = sheet.replace(" ", "_").replace("/", "_")
            short_key = clean_key(group_key, keyword)
            safe_key = short_key.replace(" ", "_").replace("(", "").replace(")", "")
            if len(sheet_names) == 1:
                out_name = f"{keyword}_{safe_key}_merged.csv"
            else:
                out_name = f"{keyword}_{safe_key}_{safe_sheet}_merged.csv"

            out_path = folder / out_name
            merged.to_csv(out_path, index=False, encoding="utf-8-sig")
            print(f"    [OK] {out_name}  ({len(merged)}행)")


def merge_csv_folder(folder: Path):
    """csv 파일 기반 병합 (아이스크림콘 방식 — 시트가 이미 파일로 분리됨)"""
    files = [f for f in folder.glob("*.csv") if "merged" not in f.name]
    if not files:
        return

    groups: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        key = strip_date(f.stem)
        groups[key].append(f)

    keyword = folder.name
    for group_key, group_files in sorted(groups.items()):
        group_files = sorted(group_files)
        print(f"\n  [{group_key}] {len(group_files)}개 파일")

        dfs = []
        for f in group_files:
            df = pd.read_csv(f, header=HEADER_ROW, encoding="utf-8-sig")
            df = df.dropna(how="all")
            # 탭 prefix 제거 (썸트렌드 CSV 특성)
            df = df.apply(lambda col: col.map(lambda x: str(x).strip() if isinstance(x, str) else x))
            dfs.append(df)

        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.drop_duplicates(subset=[merged.columns[0]])
        merged = apply_fill(merged)

        short_key = clean_key(group_key, keyword)
        safe_key = short_key.replace(" ", "_").replace("(", "").replace(")", "")
        out_name = f"{keyword}_{safe_key}_merged.csv"
        out_path = folder / out_name
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"    [OK] {out_name}  ({len(merged)}행)")


def main():
    if len(sys.argv) < 2:
        print("사용법: python merge_sometrend.py <키워드_폴더_경로>")
        sys.exit(1)

    folder = Path(sys.argv[1]).resolve()
    if not folder.exists():
        print(f"[ERROR] 폴더 없음: {folder}")
        sys.exit(1)

    print(f"폴더: {folder.name}")

    has_xlsx = any(folder.glob("*.xlsx"))
    has_csv = any(f for f in folder.glob("*.csv") if "merged" not in f.name)

    if has_xlsx:
        print("\n[xlsx 모드]")
        merge_xlsx_folder(folder)
    elif has_csv:
        print("\n[csv 모드]")
        merge_csv_folder(folder)
    else:
        print("[SKIP] 처리할 파일 없음")

    print("\n병합 완료.")


if __name__ == "__main__":
    main()
