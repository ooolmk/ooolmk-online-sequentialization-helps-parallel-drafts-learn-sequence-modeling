#!/usr/bin/env python
"""筛选 Open-PerfectBlend 的严格单轮问答数据并写成 Parquet。

保留条件：
  - conversations 恰好包含两条消息；
  - 角色顺序为 human、gpt；
  - 两条消息的 value 都是非空字符串。

输出继续使用 data/train-*.parquet 的目录布局，因此可以使用原数据集的
load_dataset(\"parquet\", data_files={\"train\": \".../data/train-*.parquet\"})
方式加载。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_P = Path(__file__).resolve().parents[1]
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT_DIR = _P / "open_perfectblend"
DEFAULT_OUTPUT_DIR = _P / "open_perfectblend_1trun"
EXPECTED_ROLES = ("human", "gpt")


def find_parquet_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix != ".parquet":
            raise ValueError(f"不是 Parquet 文件: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    files = sorted(input_path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"输入目录中没有 Parquet 文件: {input_path}")
    return files


def is_single_turn(conversation: Any) -> bool:
    if not isinstance(conversation, list) or len(conversation) != 2:
        return False

    for message, expected_role in zip(conversation, EXPECTED_ROLES):
        if not isinstance(message, dict):
            return False
        if message.get("from") != expected_role:
            return False
        if not isinstance(message.get("value"), str) or not message["value"].strip():
            return False
    return True


def filter_file(
    input_file: Path, output_file: Path, batch_size: int
) -> tuple[int, int]:
    parquet_file = pq.ParquetFile(input_file)
    schema = parquet_file.schema_arrow
    total = 0
    kept = 0

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(
        output_file,
        schema=schema,
        compression="snappy",
        use_dictionary=True,
    ) as writer:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            conversations = batch.column("conversations").to_pylist()
            mask = [is_single_turn(conversation) for conversation in conversations]
            total += len(mask)
            kept += sum(mask)

            if kept == total:
                writer.write_batch(batch)
                continue

            if not any(mask):
                continue

            table = pa.Table.from_batches([batch], schema=schema)
            writer.write_table(table.filter(pa.array(mask)))

    return total, kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"原始数据集目录或 Parquet 文件（默认：{DEFAULT_INPUT_DIR}）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认：{DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="每批读取的行数（默认：8192）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"输出目录已存在，为避免覆盖已有数据而停止: {args.output}"
        )

    input_files = find_parquet_files(args.input)
    output_data_dir = args.output / "data"
    output_data_dir.mkdir(parents=True)

    file_stats = []
    total = 0
    kept = 0
    for input_file in input_files:
        output_file = output_data_dir / input_file.name
        file_total, file_kept = filter_file(input_file, output_file, args.batch_size)
        total += file_total
        kept += file_kept
        file_stats.append(
            {
                "input": str(input_file),
                "output": str(output_file),
                "total": file_total,
                "kept": file_kept,
            }
        )
        print(f"{input_file.name}: {file_kept:,}/{file_total:,}")

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "files": file_stats,
        "total": total,
        "kept": kept,
        "removed": total - kept,
    }
    summary_file = args.output / "filter_summary.json"
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"总计保留: {kept:,}/{total:,}")
    print(f"已写入: {args.output}")
    print(f"过滤摘要: {summary_file}")
    return 0


if __name__ == "__main__":
    main()
