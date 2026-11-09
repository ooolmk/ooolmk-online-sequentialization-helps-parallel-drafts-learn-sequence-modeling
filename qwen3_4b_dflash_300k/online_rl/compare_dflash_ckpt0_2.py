#!/usr/bin/env python
"""DFlash ckpt0-2 comparison: hybrid online EAL+KL vs offline baseline.

Reads the vLLM acceptance results and the per-position waste statistics shipped
with this release, then writes a markdown table. Only DFlash checkpoints 0-2
are reported; later epochs and other draft architectures are out of scope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "qwen3_4b_dflash_300k" / "test_data" / "results"
WASTE = PROJECT / "results_dflash_ckpt0_2" / "waste"
DEFAULT_OUTPUT = PROJECT / "results_dflash_ckpt0_2" / "comparison_ckpt0_2.md"

MODELS = {
    "dflash_hybrid": {
        "acceptance": lambda k: RESULTS / f"dflash_hybrid_epoch{k}" / "results.json",
        "waste": lambda k: WASTE / f"hybrid_epoch{k}.json",
    },
    "dflash_offline": {
        "acceptance": lambda k: RESULTS / f"dflash_offline_ckpt{k}" / "results.json",
        "waste": lambda k: WASTE / f"offline_epoch{k}.json",
    },
}


def acceptance(path: Path) -> dict:
    data = json.loads(path.read_text())
    drafts = accepted = draft_tokens = 0
    per_position = [0.0] * 7
    per_dataset = {}
    for name, payload in data["datasets"].items():
        spec = payload["speculative_decoding"]
        drafts += spec["num_drafts"]
        accepted += spec["num_accepted_tokens"]
        draft_tokens += spec["num_draft_tokens"]
        per_dataset[name] = spec["acceptance_length"]
        positions = spec.get("accepted_tokens_per_position", {})
        for i in range(7):
            per_position[i] += float(positions.get(str(i), 0))
    return {
        "acceptance_length": 1.0 + accepted / drafts,
        "acceptance_rate": accepted / draft_tokens,
        "per_position": [x / drafts for x in per_position],
        "per_dataset": per_dataset,
    }


def waste(path: Path) -> dict:
    data = json.loads(path.read_text())
    counts = data["counts"]
    valid = sum(counts["valid"])
    correct = sum(counts["off_correct"])
    prefix_ok = sum(counts["off_prefix_ok"])
    wasted = sum(counts["off_wasted"])
    return {
        "reach": prefix_ok / valid,
        "offline_correct": correct / valid,
        "p_waste": wasted / valid,
        "waste_per_correct": wasted / correct,
        "pos6_waste_per_correct": counts["off_wasted"][6] / counts["off_correct"][6],
        "per_position": [
            (
                counts["off_wasted"][i] / counts["off_correct"][i]
                if counts["off_correct"][i]
                else 0.0
            )
            for i in range(7)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    lines = [
        "# DFlash online-hybrid vs offline baseline (ckpt0-2)",
        "",
        "Acceptance is measured with vLLM 0.28 on GSM8K/HumanEval/MT-Bench "
        "(temperature 0, 1,563 prompts). `w/c` is the fraction of offline-correct "
        "draft tokens that cannot be accepted because an earlier prefix token is "
        "wrong. `cond` is the conditional acceptance rate P(pos i | pos i-1).",
        "",
    ]
    for epoch in args.epochs:
        lines += [f"## Epoch {epoch}", ""]
        rows = {}
        for name, paths in MODELS.items():
            acc_path = paths["acceptance"](epoch)
            waste_path = paths["waste"](epoch)
            acc = acceptance(acc_path) if acc_path.is_file() else None
            wst = waste(waste_path) if waste_path.is_file() else None
            rows[name] = (acc, wst)
            if acc is None:
                lines.append(f"- `{name}` acceptance missing: {acc_path}")
            if wst is None:
                lines.append(f"- `{name}` waste missing: {waste_path}")
        lines += [
            "",
            "| model | AL | AR | reach | offline correct | P(waste) | waste/correct | pos6 waste/correct |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, (acc, wst) in rows.items():
            if acc is None:
                continue
            wc = f"{wst['waste_per_correct']:.4f}" if wst else "-"
            p6 = f"{wst['pos6_waste_per_correct']:.4f}" if wst else "-"
            reach = f"{wst['reach']:.4f}" if wst else "-"
            oc = f"{wst['offline_correct']:.4f}" if wst else "-"
            pw = f"{wst['p_waste']:.4f}" if wst else "-"
            lines.append(
                f"| {name} | {acc['acceptance_length']:.4f} | "
                f"{acc['acceptance_rate'] * 100:.2f}% | {reach} | {oc} | "
                f"{pw} | {wc} | {p6} |"
            )
        lines.append("")
        lines += [
            "| pos | " + " | ".join(f"pos{i}" for i in range(7)) + " |",
            "|---|" + "---|" * 7,
        ]
        for name, (acc, wst) in rows.items():
            if acc is None:
                continue
            acc_vals = " | ".join(f"{x:.3f}" for x in acc["per_position"])
            lines.append(f"| {name} acc | {acc_vals} |")
            cond = []
            for i, x in enumerate(acc["per_position"]):
                prev = acc["per_position"][i - 1] if i else 1.0
                cond.append(f"{x / prev:.3f}" if prev > 0 else "-")
            lines.append(f"| {name} cond | " + " | ".join(cond) + " |")
            if wst is not None:
                w_vals = " | ".join(f"{x:.3f}" for x in wst["per_position"])
                lines.append(f"| {name} w/c | {w_vals} |")
        lines.append("")

    text = "\n".join(lines) + "\n"
    print(text)
    Path(args.output).write_text(text)
    print(f"written {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
