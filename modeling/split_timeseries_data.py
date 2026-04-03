"""
시계열 데이터 분할 유틸리티

dataset_v2.csv를 시계열 순으로 train/valid/test 세트로 나누는 기능을 제공합니다.

사용법:
    # 기본 비율 (70% train, 15% valid, 15% test)
    python modeling/split_timeseries_data.py

    # 커스텀 비율
    python modeling/split_timeseries_data.py --train_ratio 0.8 --valid_ratio 0.1 --test_ratio 0.1

    # 분할된 데이터 저장
    python modeling/split_timeseries_data.py --save_splits
"""

import argparse
import os
import pandas as pd
import sys

# 프로젝트 루트 경로
_ROOT = os.path.dirname(os.path.dirname(__file__))
_MODELING_DIR = os.path.dirname(__file__)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _MODELING_DIR)

from modeling.data_loader import load_csv


def split_timeseries_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    date_col: str = "date"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    시계열 데이터를 시간 순서대로 train/valid/test 세트로 나눕니다.

    Parameters
    ----------
    df : pd.DataFrame
        시계열 데이터
    train_ratio : float
        학습 데이터 비율 (0~1)
    valid_ratio : float
        검증 데이터 비율 (0~1)
    test_ratio : float
        테스트 데이터 비율 (0~1)
    date_col : str
        날짜 컬럼명

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, valid_df, test_df)
    """
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1.0")

    # 날짜로 정렬
    df = df.sort_values(date_col).reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()

    return train_df, valid_df, test_df


def print_split_info(train_df, valid_df, test_df, date_col="date"):
    """분할 결과를 출력합니다."""
    total = len(train_df) + len(valid_df) + len(test_df)

    print("📊 시계열 데이터 분할 결과")
    print("=" * 50)
    print(f"전체 데이터: {total:,} 행")
    print()

    print("분할 비율:")
    print(f"  Train: {len(train_df):,} 행 ({len(train_df)/total*100:.1f}%)")
    print(f"  Valid: {len(valid_df):,} 행 ({len(valid_df)/total*100:.1f}%)")
    print(f"  Test:  {len(test_df):,} 행 ({len(test_df)/total*100:.1f}%)")
    print()

    print("시간 범위:")
    print(f"  Train: {train_df[date_col].min()} ~ {train_df[date_col].max()}")
    print(f"  Valid: {valid_df[date_col].min()} ~ {valid_df[date_col].max()}")
    print(f"  Test:  {test_df[date_col].min()} ~ {test_df[date_col].max()}")
    print()

    # 키워드 수 확인
    train_keywords = train_df['keyword'].nunique()
    valid_keywords = valid_df['keyword'].nunique()
    test_keywords = test_df['keyword'].nunique()

    print("키워드 수:")
    print(f"  Train: {train_keywords}개")
    print(f"  Valid: {valid_keywords}개")
    print(f"  Test:  {test_keywords}개")


def save_splits(train_df, valid_df, test_df, output_dir="modeling/data/splits"):
    """분할된 데이터를 별도 파일로 저장합니다."""
    os.makedirs(output_dir, exist_ok=True)

    # 저장 옵션: float 정밀도 제한으로 파일 크기 최적화
    csv_options = {
        'index': False,
        'float_format': '%.6f',  # float 정밀도 제한
        'date_format': '%Y-%m-%d'  # 날짜 형식 통일
    }

    print(f"\n💾 분할 데이터 저장 중... (대용량 파일로 시간이 걸릴 수 있습니다)")

    train_df.to_csv(os.path.join(output_dir, "train.csv"), **csv_options)
    print(f"  ✓ train.csv 저장 완료 ({len(train_df)}행)")

    valid_df.to_csv(os.path.join(output_dir, "valid.csv"), **csv_options)
    print(f"  ✓ valid.csv 저장 완료 ({len(valid_df)}행)")

    test_df.to_csv(os.path.join(output_dir, "test.csv"), **csv_options)
    print(f"  ✓ test.csv 저장 완료 ({len(test_df)}행)")

    print(f"\n💾 분할 데이터 저장 완료: {output_dir}/")
    print("  - train.csv")
    print("  - valid.csv")
    print("  - test.csv")


def main():
    parser = argparse.ArgumentParser(
        description="dataset_v2.csv를 시계열 순으로 train/valid/test 분할"
    )
    parser.add_argument(
        "--data", type=str, default="modeling/data/dataset_v2.csv",
        help="입력 데이터 파일 경로"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7,
        help="학습 데이터 비율 (기본: 0.7)"
    )
    parser.add_argument(
        "--valid_ratio", type=float, default=0.15,
        help="검증 데이터 비율 (기본: 0.15)"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.15,
        help="테스트 데이터 비율 (기본: 0.15)"
    )
    parser.add_argument(
        "--save_splits", action="store_true",
        help="분할된 데이터를 파일로 저장"
    )
    parser.add_argument(
        "--output_dir", type=str, default="modeling/data/splits",
        help="분할 데이터 저장 디렉토리"
    )

    args = parser.parse_args()

    # 비율 검증
    if abs(args.train_ratio + args.valid_ratio + args.test_ratio - 1.0) > 1e-6:
        print("❌ 오류: train_ratio + valid_ratio + test_ratio = 1.0이어야 합니다")
        return

    # 데이터 로드
    print(f"📂 데이터 로드: {args.data}")
    df = load_csv(args.data)
    print(f"   로드 완료: {len(df):,} 행 × {len(df.columns)} 열")
    print()

    # 시계열 분할
    train_df, valid_df, test_df = split_timeseries_data(
        df,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio
    )

    # 결과 출력
    print_split_info(train_df, valid_df, test_df)

    # 파일 저장 (옵션)
    if args.save_splits:
        save_splits(train_df, valid_df, test_df, args.output_dir)


if __name__ == "__main__":
    main()