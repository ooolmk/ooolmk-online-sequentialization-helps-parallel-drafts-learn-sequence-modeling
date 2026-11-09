#!/usr/bin/env python3
"""Regenerate assistant responses for a local Parquet/JSONL dataset.

The implementation reuses Speculators' official response-regeneration helpers,
but replaces its preset-only dataset loader with a local conversations loader.
The output is pre-tokenized JSONL and can be passed directly to prepare_data.py.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import aiohttp
from datasets import load_dataset
from tqdm import tqdm

REGEN_DIR = (
    Path(__file__).resolve().parent
    / "speculators-src"
    / "scripts"
    / "response_regeneration"
)
sys.path.insert(0, str(REGEN_DIR))

import script as regen  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate responses for a local conversations dataset."
    )
    parser.add_argument("--data", required=True, help="Local Parquet or JSONL path")
    parser.add_argument(
        "--model", required=True, help="Model name exposed by the vLLM server"
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/v1/chat/completions",
        help="vLLM Chat Completions endpoint",
    )
    parser.add_argument("--outfile", required=True, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--sampling-params",
        default='{"temperature": 0.7, "top_p": 0.8, "top_k": 20}',
        help="JSON sampling parameters passed to vLLM",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--resume", action="store_true", help="Skip rows already in outfile"
    )
    args = parser.parse_args()
    try:
        args.sampling_params = json.loads(args.sampling_params)
    except json.JSONDecodeError as exc:
        parser.error(f"--sampling-params is not valid JSON: {exc}")
    if not isinstance(args.sampling_params, dict):
        parser.error("--sampling-params must be a JSON object")
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    return args


def load_local_dataset(path: str):
    """Load a local dataset as a streaming HF dataset."""
    data_path = Path(path)
    if data_path.suffix == ".parquet":
        return load_dataset(
            "parquet",
            data_files={"train": str(data_path)},
            split="train",
            streaming=True,
        )
    if data_path.suffix in {".json", ".jsonl"}:
        return load_dataset(
            "json",
            data_files=str(data_path),
            split="train",
            streaming=True,
        )
    raise ValueError("--data must point to a local .parquet, .json, or .jsonl file")


async def main() -> None:
    args = parse_args()
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    error_outfile = outfile.with_name(f"{outfile.stem}.errors{outfile.suffix}")

    dataset = load_local_dataset(args.data)
    seen_ids = regen.load_seen(str(outfile)) if args.resume else set()
    detokenize = regen.build_detokenizer(args.model)

    worker_args = SimpleNamespace(
        model=args.model,
        max_tokens=args.max_tokens,
        sampling_params=args.sampling_params,
        max_retries=args.max_retries,
    )
    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=90, sock_read=None)
    connector = aiohttp.TCPConnector(
        limit=None, force_close=False, enable_cleanup_closed=True
    )
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, headers=headers
    ) as session:
        with (
            outfile.open("a", encoding="utf-8") as out_fh,
            error_outfile.open("a", encoding="utf-8") as err_fh,
            tqdm(
                total=args.limit,
                desc="Regenerating responses",
                unit="sample",
                dynamic_ncols=True,
            ) as progress,
        ):
            stats = {
                "ok": 0,
                "errors": 0,
                "truncated": 0,
                "requests": 0,
                "completion_tokens": 0,
                "total_request_s": 0.0,
                "start_time": __import__("time").perf_counter(),
            }
            workers = [
                asyncio.create_task(
                    regen.worker(
                        session,
                        queue,
                        worker_args,
                        out_fh,
                        err_fh,
                        args.endpoint,
                        progress,
                        stats,
                        detokenize,
                    )
                )
                for _ in range(args.concurrency)
            ]

            processed = 0
            for index, row in enumerate(dataset):
                if args.limit is not None and processed >= args.limit:
                    break

                turns, tool_results = regen.extract_conversation(row, None)
                if not turns:
                    continue

                primary_id = regen._primary_identifier(row)
                if primary_id in seen_ids:
                    continue

                try:
                    tools = regen.extract_tools(row)
                except ValueError as exc:
                    err_fh.write(
                        json.dumps(
                            {
                                "id": primary_id,
                                "metadata": {
                                    "idx": index,
                                    "error": repr(exc),
                                    "generations_completed": 0,
                                    "endpoint": args.endpoint,
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    err_fh.flush()
                    stats["errors"] += 1
                    progress.update(1)
                    processed += 1
                    continue

                await queue.put(
                    {
                        "idx": index,
                        "primary_id": primary_id,
                        "turns": turns,
                        "tools": tools,
                        "tool_results": tool_results,
                    }
                )
                processed += 1

            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers)
            regen._log_summary(stats)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
