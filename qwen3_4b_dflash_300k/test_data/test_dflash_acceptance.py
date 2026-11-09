#!/usr/bin/env python3
"""Benchmark Qwen3 DFlash speculative decoding on the local test datasets.

The script is intentionally self-contained so that the same command can be used
for checkpoint comparisons later.  It uses the verifier's Qwen3 tokenizer to
render prompts and to re-tokenize streamed output.  Speculative acceptance is
read from vLLM's Prometheus counters instead of estimating it from SSE chunks.

By default the script launches vLLM with checkpoint 3, tests 20 examples per
dataset, and writes JSON/JSONL results below ``test_data/results``.  Pass
``--max-samples 0`` to use every row in each dataset, or ``--no-launch-server``
to benchmark an already-running compatible vLLM endpoint.

Example (checkpoint 3, short smoke run):

    source ../../speculators_env.sh
    "$SPECULATORS_PYTHON" test_dflash_acceptance.py --max-samples 5

The launched server is expected to serve the speculator checkpoint directly;
its ``config.json`` supplies the verifier path and DFlash configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import pyarrow.parquet as pq
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR
PROJECT_ROOT = SCRIPT_DIR.parents[1]
CHECKPOINT_ROOT = SCRIPT_DIR.parent / "checkpoints"
DEFAULT_DRAFT = CHECKPOINT_ROOT / "3"
DEFAULT_TOKENIZER = PROJECT_ROOT / "Qwen3-4B-Instruct"
DEFAULT_RESULTS_ROOT = DATA_ROOT / "results"

DATASET_FILES = {
    "gsm8k": DATA_ROOT / "gsm8k" / "test-00000-of-00001.parquet",
    "humaneval": DATA_ROOT / "humaneval" / "test-00000-of-00001.parquet",
    # The source file is named train-*.parquet, but it is the mt-bench file in
    # this directory and is intentionally included in the acceptance benchmark.
    "mt-bench": DATA_ROOT / "mt-bench" / "train-00000-of-00001.parquet",
}


@dataclass(frozen=True)
class Prompt:
    dataset: str
    index: int
    sample_id: str
    prompt: str
    prompt_tokens: int


@dataclass
class RequestResult:
    dataset: str
    index: int
    sample_id: str
    success: bool
    prompt_tokens: int
    output_tokens: int = 0
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_latency_ms: float | None = None
    error: str = ""
    generated_text: str | None = None


def _apply_qwen_chat_template(tokenizer, messages: list[dict[str, str]]) -> str:
    """Render a Qwen3 prompt without enabling thinking traces."""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Keep compatibility with tokenizer versions whose template does not
        # expose enable_thinking, while still using the verifier tokenizer.
        kwargs.pop("enable_thinking")
        rendered = tokenizer.apply_chat_template(messages, **kwargs)
    if not isinstance(rendered, str):
        raise TypeError(f"Expected a rendered string, got {type(rendered).__name__}")
    return rendered


def _token_count(tokenizer, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
    ids = encoded["input_ids"]
    # Fast tokenizers return a flat list for a string; tolerate a one-item batch.
    return len(ids[0]) if ids and isinstance(ids[0], list) else len(ids)


def load_prompts(
    tokenizer,
    dataset: str,
    max_samples: int,
    seed: int,
    mt_bench_turn: str,
    shuffle: bool,
) -> list[Prompt]:
    path = DATASET_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(path)
    rows = pq.read_table(path).to_pylist()
    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    if max_samples > 0:
        order = order[:max_samples]

    prompts: list[Prompt] = []
    for index in order:
        row = rows[index]
        if dataset == "gsm8k":
            messages = [{"role": "user", "content": row["question"]}]
            sample_id = str(index)
        elif dataset == "humaneval":
            messages = [{"role": "user", "content": row["prompt"]}]
            sample_id = row["task_id"]
        else:
            turns = row["turns"]
            if not turns:
                raise ValueError(f"Empty turns for mt-bench row {index}")
            turn_index = 0 if mt_bench_turn == "first" else 1
            if turn_index >= len(turns):
                raise ValueError(
                    f"mt-bench row {index} has {len(turns)} turns; "
                    f"cannot select {mt_bench_turn}"
                )
            # The reference harness tests one user turn per request.  Keeping
            # the turn isolated avoids fabricating an assistant answer between
            # the two user turns in the raw MT-Bench record.
            messages = [
                {"role": turns[turn_index]["role"], "content": turns[turn_index]["content"]}
            ]
            sample_id = str(row["question_id"])

        rendered = _apply_qwen_chat_template(tokenizer, messages)
        prompts.append(
            Prompt(
                dataset=dataset,
                index=index,
                sample_id=sample_id,
                prompt=rendered,
                prompt_tokens=_token_count(tokenizer, rendered),
            )
        )
    return prompts


def _completion_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/v1/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/completions"
    return f"{url}/v1/completions"


def _server_root(completion_url: str) -> str:
    suffix = "/v1/completions"
    if completion_url.endswith(suffix):
        return completion_url[: -len(suffix)]
    return completion_url.rsplit("/", 1)[0]


def _fetch_text(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read().decode("utf-8", errors="replace")


def get_served_model_id(server_root: str, fallback: str) -> str:
    """Return the model id advertised by the OpenAI-compatible server."""
    try:
        payload = json.loads(_fetch_text(f"{server_root}/v1/models", timeout=10.0))
        models = payload.get("data") or []
        if models and models[0].get("id"):
            return str(models[0]["id"])
    except (OSError, urllib.error.URLError, json.JSONDecodeError, AttributeError):
        pass
    return fallback


def wait_for_health(
    health_url: str,
    process: subprocess.Popen | None,
    timeout: float,
    log_path: Path | None = None,
) -> None:
    deadline = time.monotonic() + timeout
    last_error = ""
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            suffix = f"; see {log_path}" if log_path is not None else ""
            raise RuntimeError(
                f"vLLM exited with code {process.returncode} before becoming ready{suffix}"
            )
        try:
            _fetch_text(health_url, timeout=2.0)
            return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
            time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for {health_url}: {last_error}")


def _vllm_env(vllm_python: str) -> dict[str, str]:
    env = os.environ.copy()
    # torchcodec is imported by vLLM's multimodal module in this environment.
    # Add the same CUDA library locations as start_qwen3_vllm.sh when present.
    python_path = Path(vllm_python).resolve()
    vllm_root = python_path.parents[1]
    lib_dirs = [
        vllm_root / "lib/python3.12/site-packages/nvidia/cu13/lib",
        vllm_root / "lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib",
        vllm_root / "lib",
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    additions = ":".join(str(p) for p in lib_dirs if p.exists())
    env["LD_LIBRARY_PATH"] = ":".join(x for x in (additions, existing) if x)
    return env


def launch_vllm(args: argparse.Namespace, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    vllm_python = args.vllm_python or os.environ.get("VLLM_PYTHON", sys.executable)
    cmd = [
        vllm_python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(args.draft_model),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--no-enable-chunked-prefill",
    ]
    if args.gpu_memory_utilization is not None:
        cmd.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])

    log_file = log_path.open("w", encoding="utf-8")
    env = _vllm_env(vllm_python)
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    process = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    # Keep the file descriptor attached to the child; it is closed when the
    # child exits.  The parent does not need to keep it open.
    log_file.close()
    print(f"Launched vLLM (pid={process.pid})")
    print("  " + " ".join(cmd))
    return process


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=20)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


_PROM_LINE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[0-9.eE+\-]+)(?:\s+\d+)?$"
)
_POSITION = re.compile(r'(?:^|,)position="(\d+)"(?:,|$)')


def scrape_spec_metrics(metrics_url: str) -> dict[str, Any]:
    """Read cumulative vLLM speculative-decoding counters."""
    try:
        raw = _fetch_text(metrics_url, timeout=10.0)
    except (OSError, urllib.error.URLError) as exc:
        return {"available": False, "error": str(exc), "counters": {}, "per_pos": {}}

    counters: dict[str, float] = {}
    per_pos: dict[int, float] = {}
    found = False
    for line in raw.splitlines():
        match = _PROM_LINE.match(line.strip())
        if match is None:
            continue
        name = match.group("name")
        if not name.startswith("vllm:spec_decode"):
            continue
        found = True
        if name.endswith("_total"):
            name = name[: -len("_total")]
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        if name.endswith("num_accepted_tokens_per_pos"):
            labels = match.group("labels") or ""
            pos = _POSITION.search(labels)
            if pos:
                per_pos[int(pos.group(1))] = per_pos.get(int(pos.group(1)), 0.0) + value
        elif name in {
            "vllm:spec_decode_num_drafts",
            "vllm:spec_decode_num_draft_tokens",
            "vllm:spec_decode_num_accepted_tokens",
        }:
            counters[name] = counters.get(name, 0.0) + value

    required = {
        "vllm:spec_decode_num_drafts",
        "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens",
    }
    available = required.issubset(counters)
    return {"available": available, "counters": counters, "per_pos": per_pos}


def diff_spec_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    names = [
        "vllm:spec_decode_num_drafts",
        "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens",
    ]
    counters = {
        name: after.get("counters", {}).get(name, 0.0)
        - before.get("counters", {}).get(name, 0.0)
        for name in names
    }
    per_pos_keys = set(before.get("per_pos", {})) | set(after.get("per_pos", {}))
    per_pos = {
        str(pos): after.get("per_pos", {}).get(pos, 0.0)
        - before.get("per_pos", {}).get(pos, 0.0)
        for pos in sorted(per_pos_keys)
    }
    drafts = counters[names[0]]
    draft_tokens = counters[names[1]]
    accepted = counters[names[2]]
    return {
        "available": bool(before.get("available") and after.get("available")),
        "num_drafts": int(round(drafts)),
        "num_draft_tokens": int(round(draft_tokens)),
        "num_accepted_tokens": int(round(accepted)),
        "acceptance_rate": accepted / draft_tokens if draft_tokens > 0 else None,
        # vLLM's conventional acceptance length includes the verifier bonus token.
        "acceptance_length": 1.0 + accepted / drafts if drafts > 0 else None,
        "accepted_tokens_per_position": per_pos,
        "metrics_error_before": before.get("error"),
        "metrics_error_after": after.get("error"),
    }


async def request_one(
    session: aiohttp.ClientSession,
    completion_url: str,
    model_id: str,
    prompt: Prompt,
    tokenizer,
    max_tokens: int,
    temperature: float,
    save_text: bool,
) -> RequestResult:
    payload = {
        "model": model_id,
        "prompt": prompt.prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "stream": True,
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    pieces: list[str] = []
    try:
        async with session.post(completion_url, json=payload) as response:
            if response.status != 200:
                body = await response.text()
                return RequestResult(
                    dataset=prompt.dataset,
                    index=prompt.index,
                    sample_id=prompt.sample_id,
                    success=False,
                    prompt_tokens=prompt.prompt_tokens,
                    error=f"HTTP {response.status}: {body[:1000]}",
                )
            line_buffer = ""
            async for chunk in response.content.iter_any():
                line_buffer += chunk.decode("utf-8", errors="replace")
                lines = line_buffer.split("\n")
                line_buffer = lines.pop()
                for line in lines:
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_text = line[5:].strip()
                    if not data_text or data_text == "[DONE]":
                        continue
                    try:
                        data = json.loads(data_text)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    piece = (
                        delta.get("content")
                        or delta.get("reasoning_content")
                        or choice.get("text")
                        or ""
                    )
                    if piece:
                        now = time.perf_counter()
                        if first_token_at is None:
                            first_token_at = now
                        pieces.append(piece)
            # Usually the server terminates each SSE event with a newline, but
            # process a final unterminated line as well.
            if line_buffer.strip().startswith("data:"):
                data_text = line_buffer.strip()[5:].strip()
                if data_text and data_text != "[DONE]":
                    try:
                        data = json.loads(data_text)
                    except json.JSONDecodeError:
                        data = None
                    if data:
                        choices = data.get("choices") or []
                        if choices:
                            choice = choices[0]
                            delta = choice.get("delta") or {}
                            piece = (
                                delta.get("content")
                                or delta.get("reasoning_content")
                                or choice.get("text")
                                or ""
                            )
                            if piece:
                                now = time.perf_counter()
                                if first_token_at is None:
                                    first_token_at = now
                                pieces.append(piece)
        ended = time.perf_counter()
        text = "".join(pieces)
        output_tokens = _token_count(tokenizer, text) if text else 0
        ttft_ms = (first_token_at - started) * 1000 if first_token_at else None
        e2e_ms = (ended - started) * 1000
        tpot_ms = (
            (e2e_ms - ttft_ms) / (output_tokens - 1)
            if ttft_ms is not None and output_tokens > 1
            else None
        )
        return RequestResult(
            dataset=prompt.dataset,
            index=prompt.index,
            sample_id=prompt.sample_id,
            success=True,
            prompt_tokens=prompt.prompt_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            e2e_latency_ms=e2e_ms,
            generated_text=text if save_text else None,
        )
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
        return RequestResult(
            dataset=prompt.dataset,
            index=prompt.index,
            sample_id=prompt.sample_id,
            success=False,
            prompt_tokens=prompt.prompt_tokens,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_requests(
    prompts: list[Prompt],
    completion_url: str,
    model_id: str,
    tokenizer,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    timeout: float,
    save_text: bool,
) -> list[RequestResult]:
    connector = aiohttp.TCPConnector(limit=max(1, concurrency))
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    async with aiohttp.ClientSession(
        timeout=client_timeout, connector=connector, headers={"Content-Type": "application/json"}
    ) as session:

        async def limited(prompt: Prompt) -> RequestResult:
            async with semaphore:
                return await request_one(
                    session,
                    completion_url,
                    model_id,
                    prompt,
                    tokenizer,
                    max_tokens,
                    temperature,
                    save_text,
                )

        return list(await asyncio.gather(*(limited(p) for p in prompts)))


def _stats(values: list[float | None]) -> dict[str, float | int | None]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"mean": None, "median": None, "p99": None, "count": 0}
    ordered = sorted(clean)
    p99_index = min(len(ordered) - 1, max(0, int(0.99 * len(ordered))))
    return {
        "mean": statistics.mean(clean),
        "median": statistics.median(clean),
        "p99": ordered[p99_index],
        "count": len(clean),
    }


def summarize_requests(results: list[RequestResult]) -> dict[str, Any]:
    successful = [r for r in results if r.success]
    return {
        "requested": len(results),
        "completed": len(successful),
        "failed": len(results) - len(successful),
        "total_prompt_tokens": sum(r.prompt_tokens for r in successful),
        "total_output_tokens": sum(r.output_tokens for r in successful),
        "ttft_ms": _stats([r.ttft_ms for r in successful]),
        "tpot_ms": _stats([r.tpot_ms for r in successful]),
        "e2e_latency_ms": _stats([r.e2e_latency_ms for r in successful]),
        "errors": [r.error for r in results if r.error],
    }


def _jsonable_result(result: RequestResult) -> dict[str, Any]:
    return asdict(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure Qwen3 DFlash TTFT/TPOT/e2e latency and vLLM acceptance length."
    )
    parser.add_argument("--draft-model", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_FILES), default=list(DATASET_FILES))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Samples per dataset; 0 means all rows (default: 20).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mt-bench-turn", choices=("first", "second"), default="first")
    # Match the reference harnesses' generation cap (MAX_TOKRNS=1024).
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--base-url", default=None, help="Existing server URL; defaults to http://HOST:PORT/v1.")
    parser.add_argument("--no-launch-server", action="store_true")
    parser.add_argument("--vllm-python", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--server-timeout", type=float, default=900.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--save-text", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_samples < 0:
        raise ValueError("--max-samples must be >= 0")
    if args.max_tokens <= 0 or args.concurrency <= 0:
        raise ValueError("--max-tokens and --concurrency must be positive")
    if not args.draft_model.exists():
        raise FileNotFoundError(f"DFlash checkpoint not found: {args.draft_model}")
    if not args.tokenizer.exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")

    checkpoint_label = args.draft_model.name or "checkpoint"
    run_id = f"ckpt{checkpoint_label}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = args.output_dir or DEFAULT_RESULTS_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    server_log = output_dir / "vllm_server.log"

    resolved_vllm_python = args.vllm_python or os.environ.get("VLLM_PYTHON", sys.executable)
    try:
        vllm_version = subprocess.check_output(
            [resolved_vllm_python, "-c", "import vllm; print(vllm.__version__)"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        vllm_version = "unknown"
    eval_command = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "vllm_python": resolved_vllm_python,
        "vllm_version": vllm_version,
        "draft_model": str(args.draft_model.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
    }
    (output_dir / "eval_command.txt").write_text(
        json.dumps(eval_command, ensure_ascii=False, indent=2)
        + "\n"
        + subprocess.list2cmdline([sys.executable, *sys.argv])
        + "\n"
    )

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    prompts_by_dataset = {
        dataset: load_prompts(
            tokenizer,
            dataset,
            args.max_samples,
            args.seed,
            args.mt_bench_turn,
            args.shuffle,
        )
        for dataset in args.datasets
    }
    print("Prompt counts: " + ", ".join(f"{k}={len(v)}" for k, v in prompts_by_dataset.items()))

    base_url = args.base_url or f"http://{args.host}:{args.port}/v1"
    completion_url = _completion_url(base_url)
    server_root = _server_root(completion_url)
    health_url = f"{server_root}/health"
    metrics_url = f"{server_root}/metrics"
    server_process: subprocess.Popen | None = None
    all_request_results: list[RequestResult] = []
    per_dataset: dict[str, Any] = {}

    try:
        if not args.no_launch_server:
            server_process = launch_vllm(args, server_log)
        wait_for_health(health_url, server_process, args.server_timeout, server_log)
        print(f"vLLM ready: {server_root}")
        model_id = get_served_model_id(server_root, str(args.draft_model.resolve()))
        print(f"Served model id: {model_id}")

        # A single-token warmup removes first-request compilation/loading from
        # the measured dataset metrics and from the acceptance counters.
        first_prompt = next(iter(prompts_by_dataset.values()))[0]
        warmup = asyncio.run(
            run_requests(
                [first_prompt],
                completion_url,
                model_id,
                tokenizer,
                max_tokens=1,
                temperature=args.temperature,
                concurrency=1,
                timeout=args.request_timeout,
                save_text=False,
            )
        )
        if not warmup[0].success:
            raise RuntimeError(f"Warmup request failed: {warmup[0].error}")

        for dataset, prompts in prompts_by_dataset.items():
            before = scrape_spec_metrics(metrics_url)
            started = time.perf_counter()
            results = asyncio.run(
                run_requests(
                    prompts,
                    completion_url,
                    model_id,
                    tokenizer,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    concurrency=args.concurrency,
                    timeout=args.request_timeout,
                    save_text=args.save_text,
                )
            )
            duration_s = time.perf_counter() - started
            # Give the metrics logger a moment to publish the last iteration.
            time.sleep(0.2)
            after = scrape_spec_metrics(metrics_url)
            request_summary = summarize_requests(results)
            spec_summary = diff_spec_metrics(before, after)
            dataset_summary = {
                "dataset": dataset,
                "duration_s": duration_s,
                **request_summary,
                "speculative_decoding": spec_summary,
                # Flat names make the required metrics easy to consume.
                "mean_ttft_ms": request_summary["ttft_ms"]["mean"],
                "mean_tpot_ms": request_summary["tpot_ms"]["mean"],
                "mean_e2e_latency_ms": request_summary["e2e_latency_ms"]["mean"],
                "acceptance_length": spec_summary["acceptance_length"],
            }
            per_dataset[dataset] = dataset_summary
            all_request_results.extend(results)
            print(
                f"{dataset}: completed={dataset_summary['completed']}/{dataset_summary['requested']} "
                f"ttft={dataset_summary['mean_ttft_ms']} ms "
                f"tpot={dataset_summary['mean_tpot_ms']} ms "
                f"e2e={dataset_summary['mean_e2e_latency_ms']} ms "
                f"acceptance_length={dataset_summary['acceptance_length']}"
            )

        overall_requests = summarize_requests(all_request_results)
        # Overall acceptance counters are measured as one interval spanning all
        # datasets, excluding warmup.  Re-scrape around the whole run is not
        # possible after per-dataset baselines, so sum the independent deltas.
        overall_spec: dict[str, Any] = {
            "available": all(
                bool(per_dataset[d]["speculative_decoding"]["available"])
                for d in per_dataset
            ),
            "num_drafts": sum(per_dataset[d]["speculative_decoding"]["num_drafts"] for d in per_dataset),
            "num_draft_tokens": sum(
                per_dataset[d]["speculative_decoding"]["num_draft_tokens"] for d in per_dataset
            ),
            "num_accepted_tokens": sum(
                per_dataset[d]["speculative_decoding"]["num_accepted_tokens"] for d in per_dataset
            ),
        }
        overall_spec["acceptance_rate"] = (
            overall_spec["num_accepted_tokens"] / overall_spec["num_draft_tokens"]
            if overall_spec["num_draft_tokens"]
            else None
        )
        overall_spec["acceptance_length"] = (
            1.0 + overall_spec["num_accepted_tokens"] / overall_spec["num_drafts"]
            if overall_spec["num_drafts"]
            else None
        )

        run_result = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "draft_checkpoint": str(args.draft_model.resolve()),
                "tokenizer": str(args.tokenizer.resolve()),
                "speculator_type": "dflash",
            },
            "server": {
                "completion_url": completion_url,
                "metrics_url": metrics_url,
                "model_id": model_id,
                "launched_by_script": not args.no_launch_server,
                "server_log": str(server_log) if not args.no_launch_server else None,
            },
            "config": {
                "datasets": args.datasets,
                "max_samples_per_dataset": args.max_samples,
                "seed": args.seed,
                "shuffle": args.shuffle,
                "mt_bench_turn": args.mt_bench_turn,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "concurrency": args.concurrency,
            },
            "datasets": per_dataset,
            "overall": {
                **overall_requests,
                "mean_ttft_ms": overall_requests["ttft_ms"]["mean"],
                "mean_tpot_ms": overall_requests["tpot_ms"]["mean"],
                "mean_e2e_latency_ms": overall_requests["e2e_latency_ms"]["mean"],
                "acceptance_length": overall_spec["acceptance_length"],
                "speculative_decoding": overall_spec,
            },
        }
        result_path = output_dir / "results.json"
        requests_path = output_dir / "requests.jsonl"
        result_path.write_text(json.dumps(run_result, ensure_ascii=False, indent=2), encoding="utf-8")
        with requests_path.open("w", encoding="utf-8") as handle:
            for result in all_request_results:
                handle.write(json.dumps(_jsonable_result(result), ensure_ascii=False) + "\n")
        print(f"Results written to: {result_path}")
        print(f"Per-request results written to: {requests_path}")
        return 0
    finally:
        stop_process(server_process)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
