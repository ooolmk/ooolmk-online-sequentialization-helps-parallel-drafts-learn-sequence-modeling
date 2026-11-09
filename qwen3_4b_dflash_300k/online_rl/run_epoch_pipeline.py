"""Train one epoch, pause, run the vLLM 0.28 acceptance suite, then continue.

Training itself is delegated to ``train_from_scratch_online_eal.py`` with
``--epochs-per-run 1``.  The trainer's checkpoint/optimizer/scheduler state is
loaded on the next invocation, so the four epochs form one continuous run.

Evaluation uses ``test_dflash_acceptance.py`` with the full GSM8K / HumanEval /
MT-Bench sets at temperature 0, seed 0, concurrency 1, max_model_len 8192.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONLINE_RL = Path(__file__).resolve().parent
TRAIN_SCRIPT = ONLINE_RL / "train_from_scratch_online_eal.py"
EVAL_SCRIPT = PROJECT_ROOT / "qwen3_4b_dflash_300k" / "test_data" / "test_dflash_acceptance.py"
DEFAULT_VLLM_PYTHON = os.environ.get("VLLM_PYTHON", sys.executable)
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "qwen3_4b_dflash_300k" / "test_data" / "results"
SPECULATORS_PYTHON = os.environ.get("SPECULATORS_PYTHON", sys.executable)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument(
        "--eval-tag",
        default=None,
        help="Optional results prefix; default is "
        "from_scratch_online_eal_epoch{E}_vllm028_{date}.",
    )
    parser.add_argument(
        "--eval-date",
        default=f"{datetime.now(timezone.utc):%Y%m%d}",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-anchors", type=int, required=True)
    parser.add_argument("--verify-batch", type=int, default=32)
    parser.add_argument("--chunk-blocks", type=int, default=256)
    parser.add_argument("--train-data-ratio", type=float, default=0.999)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--checkpoint-freq", type=float, default=0.01)
    parser.add_argument(
        "--verifier-label-backend",
        choices=("per_anchor", "tree"),
        default="per_anchor",
    )
    parser.add_argument(
        "--online-loss",
        choices=("soft_eal", "log_eal"),
        default="soft_eal",
    )
    parser.add_argument("--log-eal-eps", type=float, default=1e-6)
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=0.0,
        help="Prefix-weighted forward KL coefficient (requires tree labels).",
    )
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument(
        "--log-position-diagnostics",
        action="store_true",
        help="Deprecated no-op; per-step stats are no longer logged during "
        "training. Use --epoch-diagnostics instead.",
    )
    parser.add_argument(
        "--epoch-diagnostics",
        action="store_true",
        help="After each epoch eval, run a short position diagnostic on the "
        "epoch checkpoint for comparison with the offline checkpoints.",
    )
    parser.add_argument("--diag-steps", type=int, default=40)
    parser.add_argument("--diag-start-epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--muon-lr", type=float, default=None)
    parser.add_argument("--init-draft-path", default=None)
    parser.add_argument(
        "--init-config-source",
        default=str(PROJECT_ROOT / "qwen3_4b_dflash_300k" / "checkpoints" / "3"),
    )

    parser.add_argument("--vllm-python", default=DEFAULT_VLLM_PYTHON)
    parser.add_argument("--eval-max-samples", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--eval-concurrency", type=int, default=1)
    parser.add_argument("--eval-max-model-len", type=int, default=8192)
    parser.add_argument("--eval-port", type=int, default=8001)
    parser.add_argument(
        "--eval-cuda-visible-devices",
        default=None,
        help="GPU ids for the eval vLLM server; training still uses all GPUs.",
    )
    parser.add_argument(
        "--eval-gpu-memory-utilization",
        type=float,
        default=None,
        help="vLLM --gpu-memory-utilization for eval; lower it if training "
        "memory has not been released yet.",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=1,
        help="Use torchrun with this many GPUs for the training child.",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def log(message: str, log_path: Path | None = None) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
    ):
        env.pop(key, None)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def run_streaming(command: list[str], log_path: Path) -> int:
    log("$ " + subprocess.list2cmdline(command), log_path)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=subprocess_env(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        return process.wait()


def checkpoint_complete(checkpoint_dir: Path) -> bool:
    state_file = checkpoint_dir / "training_state.json"
    if not (checkpoint_dir / "model.safetensors").is_file() or not state_file.is_file():
        return False
    state = json.loads(state_file.read_text())
    return int(state.get("local_step", -1)) == 0


def has_any_checkpoint(checkpoints_dir: Path) -> bool:
    return checkpoints_dir.is_dir() and any(
        child.is_dir() and child.name.isdigit() for child in checkpoints_dir.iterdir()
    )


def train_command(args: argparse.Namespace, resume: bool) -> list[str]:
    train_args = [
        str(TRAIN_SCRIPT),
        "--run-root",
        str(Path(args.run_root).resolve()),
        "--init-config-source",
        str(Path(args.init_config_source).resolve()),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--epochs-per-run",
        "1",
        "--train-data-ratio",
        str(args.train_data_ratio),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--max-anchors",
        str(args.max_anchors),
        "--verify-batch",
        str(args.verify_batch),
        "--chunk-blocks",
        str(args.chunk_blocks),
        "--checkpoint-freq",
        str(args.checkpoint_freq),
        "--verifier-label-backend",
        args.verifier_label_backend,
        "--online-loss",
        args.online_loss,
        "--log-eal-eps",
        str(args.log_eal_eps),
        "--kl-weight",
        str(args.kl_weight),
        "--kl-temperature",
        str(args.kl_temperature),
        "--lr",
        str(args.lr),
    ]
    if args.muon_lr is not None:
        train_args.extend(["--muon-lr", str(args.muon_lr)])
    if args.init_draft_path:
        train_args.extend(
            ["--init-draft-path", str(Path(args.init_draft_path).resolve())]
        )
    if resume:
        train_args.append("--resume")
    if args.nproc_per_node > 1:
        return [
            SPECULATORS_PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            *train_args,
        ]
    return [SPECULATORS_PYTHON, *train_args]


def eval_command(args: argparse.Namespace, checkpoint: Path, output_dir: Path) -> list[str]:
    command = [
        SPECULATORS_PYTHON,
        str(EVAL_SCRIPT),
        "--draft-model",
        str(checkpoint),
        "--vllm-python",
        args.vllm_python,
        "--max-samples",
        str(args.eval_max_samples),
        "--seed",
        str(args.eval_seed),
        "--temperature",
        "0.0",
        "--concurrency",
        str(args.eval_concurrency),
        "--max-model-len",
        str(args.eval_max_model_len),
        "--datasets",
        "gsm8k",
        "humaneval",
        "mt-bench",
        "--port",
        str(args.eval_port),
        "--output-dir",
        str(output_dir),
    ]
    if args.eval_cuda_visible_devices:
        command.extend(["--cuda-visible-devices", args.eval_cuda_visible_devices])
    if args.eval_gpu_memory_utilization is not None:
        command.extend(
            ["--gpu-memory-utilization", str(args.eval_gpu_memory_utilization)]
        )
    return command


def diagnostic_command(
    args: argparse.Namespace, checkpoint: Path, diag_root: Path
) -> list[str]:
    diag_args = [
        str(TRAIN_SCRIPT),
        "--run-root",
        str(diag_root),
        "--init-draft-path",
        str(checkpoint),
        "--init-config-source",
        str(Path(args.init_config_source).resolve()),
        "--seed",
        str(args.seed),
        "--epochs",
        "1",
        "--epochs-per-run",
        "1",
        "--max-steps",
        str(args.diag_steps),
        "--train-data-ratio",
        str(args.train_data_ratio),
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--max-anchors",
        str(args.max_anchors),
        "--verify-batch",
        str(args.verify_batch),
        "--chunk-blocks",
        str(args.chunk_blocks),
        "--checkpoint-freq",
        "1.0",
        "--verifier-label-backend",
        "tree",
        "--online-loss",
        "log_eal",
        "--log-eal-eps",
        "1e-6",
        "--kl-weight",
        "0",
        "--lr",
        "0",
        "--muon-lr",
        "0",
        "--log-position-diagnostics",
    ]
    if args.nproc_per_node > 1:
        return [
            SPECULATORS_PYTHON,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.nproc_per_node}",
            *diag_args,
        ]
    return [SPECULATORS_PYTHON, *diag_args]


def summarize_eval(epoch: int, results_dir: Path) -> dict:
    result = json.loads((results_dir / "results.json").read_text())
    overall = result.get("overall", {})
    spec = overall.get("speculative_decoding", {})
    per_dataset = {}
    for name, payload in result.get("datasets", {}).items():
        per_dataset[name] = {
            "completed": payload.get("completed"),
            "failed": payload.get("failed"),
            "acceptance_length": payload.get("acceptance_length"),
            "acceptance_rate": (payload.get("speculative_decoding") or {}).get(
                "acceptance_rate"
            ),
            "mean_e2e_latency_ms": payload.get("mean_e2e_latency_ms"),
        }
    return {
        "epoch": epoch,
        "results_dir": str(results_dir),
        "requested": overall.get("requested"),
        "completed": overall.get("completed"),
        "failed": overall.get("failed"),
        "acceptance_length": overall.get("acceptance_length"),
        "acceptance_rate": spec.get("acceptance_rate"),
        "mean_e2e_latency_ms": overall.get("mean_e2e_latency_ms"),
        "per_dataset": per_dataset,
    }


def write_comparison(run_root: Path, summaries: list[dict]) -> None:
    lines = [
        "# From-scratch online soft-EAL acceptance",
        "",
        "| model | acceptance length | acceptance rate | mean e2e latency (ms) | completed/failed |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "| online epoch{epoch} | {acceptance_length:.4f} | "
            "{acceptance_rate:.4f} | {mean_e2e_latency_ms:.1f} | "
            "{completed}/{failed} |".format(
                epoch=summary["epoch"],
                acceptance_length=summary["acceptance_length"] or 0.0,
                acceptance_rate=summary["acceptance_rate"] or 0.0,
                mean_e2e_latency_ms=summary["mean_e2e_latency_ms"] or 0.0,
                completed=summary["completed"],
                failed=summary["failed"],
            )
        )
    lines += [
        "",
        "Baseline ckpt0/1/2 full vLLM-0.28 evaluations still need to be run; "
        "ckpt3 reference is 4.8075 acceptance length / 0.5439 acceptance rate "
        "(test_data/results/ckpt3_full_vllm028_20261012).",
        "",
        "Per-dataset details:",
        "",
        "| model | dataset | acceptance length | acceptance rate | mean e2e latency (ms) |",
        "|---|---|---:|---:|---:|",
    ]
    for summary in summaries:
        for name, payload in summary["per_dataset"].items():
            lines.append(
                "| online epoch{epoch} | {name} | {length:.4f} | {rate:.4f} | "
                "{latency:.1f} |".format(
                    epoch=summary["epoch"],
                    name=name,
                    length=payload["acceptance_length"] or 0.0,
                    rate=payload["acceptance_rate"] or 0.0,
                    latency=payload["mean_e2e_latency_ms"] or 0.0,
                )
            )
    (run_root / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    results_root = Path(args.results_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)
    log_path = run_root / "pipeline.log"
    checkpoints_dir = run_root / "checkpoints"
    state_path = run_root / "pipeline_state.json"
    summaries: list[dict] = []

    log(
        f"pipeline start: epochs={args.epochs} seed={args.seed} "
        f"max_anchors={args.max_anchors} verify_batch={args.verify_batch} "
        f"eval_tag={args.eval_tag or 'default'} eval_date={args.eval_date}",
        log_path,
    )

    for epoch in range(args.epochs):
        checkpoint_dir = checkpoints_dir / str(epoch)
        if not checkpoint_complete(checkpoint_dir):
            resume = has_any_checkpoint(checkpoints_dir) or (
                args.dry_run and epoch > 0
            )
            command = train_command(args, resume=resume)
            if args.dry_run:
                log("DRY-RUN train: " + subprocess.list2cmdline(command), log_path)
            else:
                log(f"epoch {epoch}: training (resume={resume})", log_path)
                code = run_streaming(command, log_path)
                if code != 0:
                    log(f"epoch {epoch}: training failed with exit code {code}", log_path)
                    return code
            if not args.dry_run and not checkpoint_complete(checkpoint_dir):
                log(
                    f"epoch {epoch}: checkpoint is missing or mid-epoch at "
                    f"{checkpoint_dir}",
                    log_path,
                )
                return 1
            if not args.dry_run:
                _write_state(state_path, epoch, "trained", None)
        else:
            log(f"epoch {epoch}: checkpoint already complete", log_path)

        if args.skip_eval:
            continue

        if args.eval_tag:
            results_dir = results_root / f"{args.eval_tag}_epoch{epoch}"
        else:
            results_dir = (
                results_root
                / f"from_scratch_online_eal_epoch{epoch}_vllm028_{args.eval_date}"
            )
        if (results_dir / "results.json").is_file() and not args.force_eval:
            log(f"epoch {epoch}: evaluation already present at {results_dir}", log_path)
        else:
            if results_dir.exists() and any(results_dir.iterdir()):
                if not args.force_eval:
                    log(
                        f"epoch {epoch}: refusing to overwrite non-empty evaluation "
                        f"directory {results_dir}; rename it or pass --force-eval",
                        log_path,
                    )
                    return 1
                backup = results_dir.with_name(
                    results_dir.name
                    + f"_backup_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
                )
                results_dir.rename(backup)
                log(
                    f"epoch {epoch}: moved incomplete evaluation to {backup}",
                    log_path,
                )
            command = eval_command(args, checkpoint_dir, results_dir)
            if args.dry_run:
                log("DRY-RUN eval: " + subprocess.list2cmdline(command), log_path)
                continue
            log(f"epoch {epoch}: evaluating with vLLM 0.28", log_path)
            code = run_streaming(command, log_path)
            if code != 0:
                log(f"epoch {epoch}: evaluation failed with exit code {code}", log_path)
                return code
            if not (results_dir / "results.json").is_file():
                log(f"epoch {epoch}: results.json missing", log_path)
                return 1

        summary = summarize_eval(epoch, results_dir)
        summaries.append(summary)
        with (run_root / "eval_summary.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        _write_state(state_path, epoch, "evaluated", results_dir)
        log(
            "epoch {epoch}: acceptance_length={length:.4f} "
            "acceptance_rate={rate:.4f} completed={completed}/{requested} "
            "failed={failed}".format(
                epoch=summary["epoch"],
                length=summary["acceptance_length"] or 0.0,
                rate=summary["acceptance_rate"] or 0.0,
                completed=summary["completed"],
                requested=summary["requested"],
                failed=summary["failed"],
            ),
            log_path,
        )

        if args.epoch_diagnostics and epoch >= args.diag_start_epoch:
            diag_root = run_root / f"position_diag_epoch{epoch}"
            if (diag_root / "metrics.jsonl").is_file():
                log(
                    f"epoch {epoch}: position diagnostics already present at "
                    f"{diag_root}",
                    log_path,
                )
            elif diag_root.exists() and any(diag_root.iterdir()):
                log(
                    f"epoch {epoch}: refusing non-empty diagnostic dir "
                    f"{diag_root}",
                    log_path,
                )
            else:
                command = diagnostic_command(args, checkpoint_dir, diag_root)
                if args.dry_run:
                    log("DRY-RUN diag: " + subprocess.list2cmdline(command), log_path)
                else:
                    log(f"epoch {epoch}: running position diagnostics", log_path)
                    code = run_streaming(command, log_path)
                    if code != 0:
                        log(
                            f"epoch {epoch}: position diagnostics failed with "
                            f"exit code {code}",
                            log_path,
                        )
                    else:
                        log(f"epoch {epoch}: position diagnostics done", log_path)

    if summaries:
        write_comparison(run_root, summaries)
        log(f"comparison written to {run_root / 'comparison.md'}", log_path)
    log("pipeline complete", log_path)
    return 0


def _write_state(path: Path, epoch: int, stage: str, results_dir: Path | None) -> None:
    state = {
        "last_epoch": epoch,
        "stage": stage,
        "results_dir": str(results_dir) if results_dir else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "wall_time": time.time(),
    }
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
