#!/usr/bin/env python
"""保留 Open-PerfectBlend 的全部样本，并将多轮对话截取为第一轮。

处理规则：
  - 只有一条消息的样本：原样保留；
  - 只有一轮（两条消息）的样本：原样保留；
  - 多轮（超过两条消息）的样本：只保留前两条消息；
  - 不检查或修改 value 内容，因此空内容的一轮对话也会保留。

输出继续使用 data/train-*.parquet 的目录布局，可使用原数据集的
load_dataset("parquet", data_files={"train": ".../data/train-*.parquet"})
方式加载。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

_P = Path(__file__).resolve().parents[1]
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT_DIR = _P / "open_perfectblend"
DEFAULT_OUTPUT_DIR = _P / "open_perfectblend_1trun_v2"


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


def keep_first_turn(conversation: Any) -> Any:
    if isinstance(conversation, list) and len(conversation) > 2:
        return conversation[:2]
    return conversation


def length_key(conversation: Any) -> int | str:
    if isinstance(conversation, list):
        return len(conversation)
    return type(conversation).__name__


def normalize_file(
    input_file: Path, output_file: Path, batch_size: int
) -> tuple[int, Counter[int | str], Counter[int | str]]:
    parquet_file = pq.ParquetFile(input_file)
    schema = parquet_file.schema_arrow
    conversation_index = schema.get_field_index("conversations")
    conversation_type = schema.field("conversations").type
    total = 0
    input_lengths: Counter[int | str] = Counter()
    output_lengths: Counter[int | str] = Counter()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(
        output_file,
        schema=schema,
        compression="snappy",
        use_dictionary=True,
    ) as writer:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            conversations = batch.column("conversations").to_pylist()
            normalized = [keep_first_turn(conversation) for conversation in conversations]
            total += len(conversations)
            input_lengths.update(length_key(conversation) for conversation in conversations)
            output_lengths.update(length_key(conversation) for conversation in normalized)

            table = pa.Table.from_batches([batch], schema=schema)
            conversation_array = pa.array(normalized, type=conversation_type)
            table = table.set_column(
                conversation_index,
                schema.field("conversations"),
                conversation_array,
            )
            writer.write_table(table)

    return total, input_lengths, output_lengths


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
    output_data_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    input_lengths: Counter[int | str] = Counter()
    output_lengths: Counter[int | str] = Counter()
    file_stats = []
    for input_file in input_files:
        output_file = output_data_dir / input_file.name
        file_total, file_input_lengths, file_output_lengths = normalize_file(
            input_file, output_file, args.batch_size
        )
        total += file_total
        input_lengths.update(file_input_lengths)
        output_lengths.update(file_output_lengths)
        file_stats.append(
            {
                "input": str(input_file),
                "output": str(output_file),
                "rows": file_total,
                "input_lengths": dict(file_input_lengths),
                "output_lengths": dict(file_output_lengths),
            }
        )
        print(
            f"{input_file.name}: {file_total:,} rows, "
            f"input lengths={dict(sorted(file_input_lengths.items(), key=str))}, "
            f"output lengths={dict(sorted(file_output_lengths.items(), key=str))}"
        )

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "files": file_stats,
        "total": total,
        "input_lengths": dict(input_lengths),
        "output_lengths": dict(output_lengths),
        "rows_removed": 0,
    }
    summary_file = args.output / "normalization_summary.json"
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"总行数: {total:,}（未删除样本，仅截取多轮对话）")
    print(f"已写入: {args.output}")
    print(f"处理摘要: {summary_file}")
    return 0


if __name__ == "__main__":
    main()
