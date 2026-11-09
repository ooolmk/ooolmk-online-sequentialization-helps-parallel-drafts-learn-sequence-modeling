#!/usr/bin/env python
"""Per-position offline-correct/wasted statistics on the 1,563 test prompts.

Replays target + draft on the target-greedy responses saved by the vLLM eval.
Draft context uses target layers 2/18/33 (same aux layers as training);
offline labels use the target's own final norm + lm_head.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "qwen3_4b_dflash_300k" / "test_data"))
sys.path.insert(0, str(PROJECT / "speculators-src" / "src"))

from test_dflash_acceptance import load_prompts  # noqa: E402
from speculators.models.dflash.core import DFlashDraftModel  # noqa: E402

VERIFIER = PROJECT / "Qwen3-4B-Instruct"
RESULTS = PROJECT / "qwen3_4b_dflash_300k" / "test_data" / "results"
HYBRID_RUN = (
    PROJECT
    / "qwen3_4b_dflash_300k"
    / "online_rl"
    / "runs"
    / "dflash_hybrid_ddp2"
)
OFFLINE_CKPT = PROJECT / "qwen3_4b_dflash_300k" / "checkpoints"
WASTE_OUTPUT = PROJECT / "results_dflash_ckpt0_2" / "waste"
AUX_LAYERS = (2, 18, 33)
BLOCK_SIZE = 8

OFFLINE_EVAL = {
    0: "dflash_offline_ckpt0",
    1: "dflash_offline_ckpt1",
    2: "dflash_offline_ckpt2",
    3: "dflash_offline_ckpt3",
}


def acceptance(path: Path) -> dict:
    data = json.loads(path.read_text())
    drafts = accepted = draft_tokens = 0
    per_position = [0.0] * 7
    for payload in data["datasets"].values():
        spec = payload["speculative_decoding"]
        drafts += spec["num_drafts"]
        accepted += spec["num_accepted_tokens"]
        draft_tokens += spec["num_draft_tokens"]
        positions = spec.get("accepted_tokens_per_position", {})
        for i in range(7):
            per_position[i] += float(positions.get(str(i), 0))
    return {
        "acceptance_length": 1.0 + accepted / drafts,
        "acceptance_rate": accepted / draft_tokens,
        "per_position": [x / drafts for x in per_position],
    }


def load_test_texts(text_run: Path, tokenizer, limit: int) -> list[dict]:
    prompts = {}
    for dataset in ("gsm8k", "humaneval", "mt-bench"):
        for prompt in load_prompts(
            tokenizer, dataset, 0, 0, "first", True
        ):
            prompts[(dataset, prompt.index)] = prompt.prompt
    records = []
    with (text_run / "requests.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if not row.get("success") or not row.get("generated_text"):
                continue
            key = (row["dataset"], row["index"])
            prompt = prompts.get(key)
            if prompt is None:
                continue
            full_ids = tokenizer(
                prompt + row["generated_text"],
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            if isinstance(full_ids[0], list):
                full_ids = full_ids[0]
            response_start = int(row["prompt_tokens"])
            if response_start >= len(full_ids) or len(full_ids) > 8192:
                continue
            records.append(
                {
                    "dataset": row["dataset"],
                    "index": row["index"],
                    "input_ids": full_ids,
                    "response_start": response_start,
                }
            )
            if limit and len(records) >= limit:
                return records
    return records


def target_outputs(target, ids: torch.Tensor):
    with torch.inference_mode():
        out = target.model(
            input_ids=ids.unsqueeze(0),
            attention_mask=torch.ones_like(ids).unsqueeze(0),
            use_cache=False,
            output_hidden_states=True,
        )
        aux = torch.cat(
            [out.hidden_states[i][0] for i in AUX_LAYERS], dim=-1
        )
        last = out.last_hidden_state[0]
    return aux, last


def replay_checkpoint(draft, records, target, device, max_anchors):
    counts = {key: [0] * 7 for key in (
        "valid", "off_correct", "off_prefix_ok", "off_wasted"
    )}
    for record in records:
        ids = torch.tensor(record["input_ids"], device=device)
        length = ids.numel()
        response_start = record["response_start"]
        if length - response_start < BLOCK_SIZE:
            continue
        aux, last = target_outputs(target, ids)
        loss_mask = torch.zeros(1, length, dtype=torch.long, device=device)
        loss_mask[0, response_start:] = 1
        document_ids = torch.zeros(1, length, dtype=torch.long, device=device)
        position_ids = torch.arange(length, device=device).unsqueeze(0)
        anchors = max(1, min(max_anchors, (length - response_start) // BLOCK_SIZE))
        with torch.inference_mode():
            _, logits, _, aligned_mask, block_indices = draft._backbone_forward(
                hidden_states=aux.unsqueeze(0),
                input_ids=ids.unsqueeze(0),
                loss_mask=loss_mask,
                verifier_last_hidden_states=last.unsqueeze(0),
                document_ids=document_ids,
                position_ids=position_ids,
                max_anchors=anchors,
                compute_targets=False,
            )
        num_blocks = block_indices.numel() // BLOCK_SIZE
        blocks = block_indices.reshape(num_blocks, BLOCK_SIZE)
        scores = logits.reshape(num_blocks, BLOCK_SIZE, -1)[:, 1:]
        proposal = scores.argmax(-1)
        valid = aligned_mask.reshape(num_blocks, BLOCK_SIZE)[:, 1:].bool()
        valid = valid & document_ids[0][blocks[:, 1:]].eq(
            document_ids[0][blocks[:, :1]]
        )
        valid = valid.long().cumprod(-1).bool()
        positions = blocks[:, 1:] - 1
        positions = positions.clamp(0, length - 1)
        with torch.inference_mode():
            target_logits = target.lm_head(last[positions])
        labels = target_logits.argmax(-1)
        correct = proposal.eq(labels) & valid
        prefix_ok = torch.ones_like(correct)
        prefix_ok[:, 1:] = correct[:, :-1].long().cumprod(-1).bool()
        prefix_ok = prefix_ok & valid
        wasted = correct & ~prefix_ok
        for i in range(7):
            counts["valid"][i] += int(valid[:, i].sum())
            counts["off_correct"][i] += int(correct[:, i].sum())
            counts["off_prefix_ok"][i] += int(prefix_ok[:, i].sum())
            counts["off_wasted"][i] += int(wasted[:, i].sum())
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text-run",
        default=str(RESULTS / "eval_text_replay"),
    )
    parser.add_argument(
        "--output",
        default=str(WASTE_OUTPUT),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-anchors", type=int, default=128)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--hybrid-run-root", default=str(HYBRID_RUN))
    parser.add_argument("--offline-ckpt-root", default=str(OFFLINE_CKPT))
    args = parser.parse_args()
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    hybrid_run = Path(args.hybrid_run_root)
    offline_ckpt = Path(args.offline_ckpt_root)

    tokenizer = AutoTokenizer.from_pretrained(
        str(VERIFIER), local_files_only=True
    )
    records = load_test_texts(Path(args.text_run), tokenizer, args.limit)
    print(f"records: {len(records)}")
    target = (
        AutoModelForCausalLM.from_pretrained(
            str(VERIFIER),
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )

    groups = [
        ("hybrid", k, hybrid_run / "checkpoints" / str(k))
        for k in args.epochs
    ] + [
        ("offline", k, offline_ckpt / str(k)) for k in args.epochs
    ]
    report = []
    for method, epoch, checkpoint in groups:
        print(f"replay {method} epoch {epoch}: {checkpoint}", flush=True)
        draft = (
            DFlashDraftModel.from_pretrained(
                str(checkpoint),
                verifier=str(VERIFIER),
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
        counts = replay_checkpoint(
            draft, records, target, device, args.max_anchors
        )
        del draft
        torch.cuda.empty_cache()
        if method == "hybrid":
            eval_path = RESULTS / f"dflash_hybrid_epoch{epoch}" / "results.json"
        else:
            eval_path = RESULTS / OFFLINE_EVAL[epoch] / "results.json"
        acc = acceptance(eval_path)
        payload = {
            "method": method,
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "acceptance": acc,
            "counts": counts,
        }
        (output / f"{method}_epoch{epoch}.json").write_text(
            json.dumps(payload, indent=1)
        )
        report.append(payload)
    (output / "all_groups.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
