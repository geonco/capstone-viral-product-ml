# CSV 로드 + 컬럼 검증/재정렬

import os
import warnings

import pandas as pd

from pipeline.config import ALL_COLS, OPTIONAL_COLS, REQUIRED_COLS


def _read_raw(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"[data_loader] CSV 파일을 찾을 수 없습니다: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[data_loader] 로드 완료: {df.shape[0]}행 x {df.shape[1]}열  ({csv_path})")
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    present = set(df.columns)

    missing_required = [c for c in REQUIRED_COLS if c not in present]
    if missing_required:
        raise ValueError(
            f"[data_loader] 필수 컬럼이 CSV에 없습니다: {missing_required}"
        )

    missing_optional = [c for c in OPTIONAL_COLS if c not in present]
    if missing_optional:
        warnings.warn(
            f"[data_loader] 선택 컬럼이 없어 건너뜁니다: {missing_optional}",
            UserWarning,
            stacklevel=3,
        )

    extra = sorted(present - set(ALL_COLS))
    if extra:
        print(f"[data_loader] 미지 컬럼 감지 (맨 뒤에 유지됩니다): {extra}")


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    present = set(df.columns)
    canonical_present = [c for c in ALL_COLS if c in present]
    extra = [c for c in df.columns if c not in set(ALL_COLS)]
    return df[canonical_present + extra]


def load_csv(csv_path: str) -> pd.DataFrame:
    df = _read_raw(csv_path)
    _validate_columns(df)
    df = _reorder_columns(df)
    return df
