#!/usr/bin/env python
"""检查 Open-PerfectBlend 中的 conversations 是否都是单轮问答。

单轮问答的严格定义：
  1. conversations 恰好包含两条消息；
  2. 两条消息的角色顺序为 human、gpt；
  3. 两条消息的 value 都是非空字符串。

脚本按 Parquet batch 流式读取，不会将整个数据集加载到内存中。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_P = Path(__file__).resolve().parents[1]
from typing import Any, Iterable

import pyarrow.parquet as pq


DEFAULT_DATASET_DIR = _P / "open_perfectblend"
EXPECTED_ROLES = ("human", "gpt")


def parquet_files(path: Path) -> list[Path]:
    """返回待检查的 Parquet 文件，支持传入目录或单个文件。"""
    if path.is_file():
        if path.suffix != ".parquet":
            raise ValueError(f"不是 Parquet 文件: {path}")
        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"路径不存在: {path}")

    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"目录中没有找到 Parquet 文件: {path}")
    return files


def validate_conversation(conversation: Any) -> list[str]:
    """返回一条 conversation 不符合单轮问答定义的原因。"""
    reasons: list[str] = []

    if not isinstance(conversation, list):
        return [f"conversations 类型为 {type(conversation).__name__}，不是 list"]

    if len(conversation) != 2:
        return [f"消息数为 {len(conversation)}，不是 2"]

    roles: list[Any] = []
    for index, message in enumerate(conversation):
        if not isinstance(message, dict):
            reasons.append(f"第 {index} 条消息类型为 {type(message).__name__}，不是 dict")
            continue

        role = message.get("from")
        roles.append(role)
        if not isinstance(role, str):
            reasons.append(f"第 {index} 条消息的 from 不是字符串")

        value = message.get("value")
        if not isinstance(value, str):
            reasons.append(f"第 {index} 条消息的 value 不是字符串")
        elif not value.strip():
            reasons.append(f"第 {index} 条消息的 value 为空")

    if roles != list(EXPECTED_ROLES):
        reasons.append(f"角色顺序为 {roles!r}，期望 {list(EXPECTED_ROLES)!r}")

    return reasons


def message_roles(conversation: Any) -> list[Any]:
    if not isinstance(conversation, list):
        return []
    return [
        message.get("from") if isinstance(message, dict) else None
        for message in conversation
    ]


def short_value(value: Any, limit: int = 80) -> str:
    if not isinstance(value, str):
        return repr(value)
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def inspect_files(
    files: Iterable[Path], batch_size: int, max_examples: int, show_content: bool
) -> dict[str, Any]:
    total = 0
    valid = 0
    invalid = 0
    length_counts: Counter[int] = Counter()
    role_pattern_counts: Counter[tuple[Any, ...]] = Counter()
    reason_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        columns = ["conversations"]
        has_source = "source" in parquet_file.schema_arrow.names
        if has_source:
            columns.append("source")

        file_row = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            conversations = batch.column("conversations").to_pylist()
            sources = (
                batch.column("source").to_pylist()
                if has_source
                else [None] * len(conversations)
            )

            for conversation, source in zip(conversations, sources):
                total += 1
                file_row += 1
                length = len(conversation) if isinstance(conversation, list) else None
                if length is not None:
                    length_counts[length] += 1
                roles = tuple(message_roles(conversation))
                role_pattern_counts[roles] += 1

                reasons = validate_conversation(conversation)
                if reasons:
                    invalid += 1
                    for reason in reasons:
                        reason_counts[reason] += 1
                    if len(examples) < max_examples:
                        example: dict[str, Any] = {
                            "file": str(file_path),
                            "row": file_row,
                            "global_row": total,
                            "source": source,
                            "length": length,
                            "roles": list(roles),
                            "reasons": reasons,
                        }
                        if show_content and isinstance(conversation, list):
                            example["values"] = [
                                short_value(
                                    message.get("value")
                                    if isinstance(message, dict)
                                    else message
                                )
                                for message in conversation
                            ]
                        examples.append(example)
                else:
                    valid += 1

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "length_counts": length_counts,
        "role_pattern_counts": role_pattern_counts,
        "reason_counts": reason_counts,
        "examples": examples,
    }


def print_report(files: list[Path], result: dict[str, Any]) -> None:
    total = result["total"]
    valid = result["valid"]
    invalid = result["invalid"]
    percentage = valid / total * 100 if total else 0.0

    print(f"检查文件数: {len(files)}")
    print(f"总样本数: {total:,}")
    print(f"严格单轮问答: {valid:,} ({percentage:.2f}%)")
    print(f"不符合单轮问答: {invalid:,} ({100 - percentage:.2f}%)")
    print()

    print("conversation 长度分布:")
    for length, count in sorted(result["length_counts"].items()):
        print(f"  {length}: {count:,}")

    print("\n不符合原因统计:")
    for reason, count in result["reason_counts"].most_common():
        print(f"  {count:,}: {reason}")

    if result["examples"]:
        print("\n不符合样本示例:")
        for example in result["examples"]:
            print(json.dumps(example, ensure_ascii=False, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"数据集目录或 Parquet 文件（默认：{DEFAULT_DATASET_DIR}）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="每批读取的行数（默认：8192）",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="最多打印多少条不符合样本（默认：10）",
    )
    parser.add_argument(
        "--show-content",
        action="store_true",
        help="在示例中额外打印截断后的消息内容",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    if args.max_examples < 0:
        raise ValueError("--max-examples 不能小于 0")

    files = parquet_files(args.path)
    result = inspect_files(files, args.batch_size, args.max_examples, args.show_content)
    print_report(files, result)
    return 0 if result["invalid"] == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"错误: {error}", file=sys.stderr)
        sys.exit(2)
