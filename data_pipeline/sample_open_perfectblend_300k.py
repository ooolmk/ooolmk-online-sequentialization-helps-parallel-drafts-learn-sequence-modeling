#!/usr/bin/env python
"""打乱 1turn 数据集并随机抽取固定数量的样本。

输出使用 data/train-*.parquet 布局，可以继续使用：
load_dataset("parquet", data_files={"train": ".../data/train-*.parquet"})
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_P = Path(__file__).resolve().parents[1]

from datasets import load_dataset


DEFAULT_INPUT_DIR = _P / "open_perfectblend_1trun_v2"
DEFAULT_OUTPUT_DIR = _P / "open_perfectblend_1trun_v2_300k"
DEFAULT_SAMPLE_SIZE = 300_000
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"输入数据集目录（默认：{DEFAULT_INPUT_DIR}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出数据集目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"抽样数量（默认：{DEFAULT_SAMPLE_SIZE:,}）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"随机种子（默认：{DEFAULT_SEED}）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size 必须大于 0")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"输出目录已存在，为避免覆盖已有数据而停止: {args.output}"
        )
    if not (args.input / "data").is_dir():
        raise FileNotFoundError(f"输入目录缺少 data 子目录: {args.input}")

    data_files = str(args.input / "data" / "train-*.parquet")
    dataset = load_dataset(
        "parquet",
        data_files={"train": data_files},
        split="train",
        cache_dir="/tmp/open_perfectblend_1trun_v2_300k_cache",
    )
    if args.size > len(dataset):
        raise ValueError(
            f"抽样数量 {args.size:,} 大于输入数据集大小 {len(dataset):,}"
        )

    subset = dataset.shuffle(seed=args.seed).select(range(args.size))

    output_data_dir = args.output / "data"
    output_data_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_data_dir / "train-00000-of-00001.parquet"
    subset.to_parquet(str(output_file))

    source_counts = Counter(subset["source"])
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "input_rows": len(dataset),
        "output_rows": len(subset),
        "sample_size": args.size,
        "seed": args.seed,
        "source_counts": dict(source_counts),
        "output_file": str(output_file),
    }
    summary_file = args.output / "sample_summary.json"
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"输入样本数: {len(dataset):,}")
    print(f"输出样本数: {len(subset):,}")
    print(f"随机种子: {args.seed}")
    print(f"已写入: {output_file}")
    print(f"抽样摘要: {summary_file}")
    return 0


if __name__ == "__main__":
    main()
