"""From-scratch online hybrid (EAL + forward KL) training for DSpark.

This is a DSpark-specific copy of the DFlash online trainer.  The DSpark
backbone is the DFlash block-parallel draft model; the online proposal path
applies the Markov head sequentially with the previously selected token, so
the logits used by the EAL/KL loss and the verifier tree labels describe the
same draft prefix.  The confidence head is kept in the checkpoint but its BCE
term is optional via ``--confidence-head-alpha``.

One process trains at most ``--epochs-per-run`` epochs (default 1) and then
exits, so the caller can run a real acceptance evaluation between epochs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acceptance_objectives import (  # noqa: E402
    forward_kl_chunked,
    prefix_weights_logspace,
    soft_eal_loss_chunked,
)

from speculators._vendor.hs_connectors.transfer import FileTransfer  # noqa: E402
from speculators.model import SpeculatorModel  # noqa: E402
from speculators.models.dflash.config import DFlashSpeculatorConfig  # noqa: E402
from speculators.models.dspark.core import DSparkDraftModel  # noqa: E402
from speculators.train.dataloader import create_train_val_loaders  # noqa: E402
from speculators.train.distributed import (  # noqa: E402
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed,
    maybe_destroy_distributed,
    maybe_setup_distributed,
)
from speculators.train.trainer import Trainer, TrainerConfig  # noqa: E402
from verifier_tree_attention import online_verifier_labels_tree  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIN_STEP_PCT = 0.25
LOG = logging.getLogger("online-eal")


def _confidence_bce_chunked(
    scores,
    branch_hidden,
    confidence_logits,
    valid_mask,
    verifier_lm_head,
    chunk_blocks,
):
    """BCE against the tree-prefix analytical acceptance overlap."""
    if confidence_logits.shape != valid_mask.shape:
        raise ValueError("confidence logits must match the valid mask")
    total = scores.new_zeros((), dtype=torch.float32)
    count = scores.new_zeros((), dtype=torch.float32)
    for start in range(0, scores.shape[0], chunk_blocks):
        end = min(start + chunk_blocks, scores.shape[0])
        with torch.no_grad():
            hidden = branch_hidden[start:end]
            weight = getattr(verifier_lm_head, "weight", None)
            if weight is not None:
                hidden = hidden.to(dtype=weight.dtype)
            target_p = torch.softmax(
                verifier_lm_head(hidden).float(), dim=-1
            )
            draft_p = torch.softmax(scores[start:end].float(), dim=-1)
            overlap = torch.minimum(draft_p, target_p).sum(dim=-1)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            confidence_logits[start:end].float(),
            overlap,
            reduction="none",
        )
        mask = valid_mask[start:end].to(bce.dtype)
        total = total + (bce * mask).sum()
        count = count + mask.sum()
    return total / count.clamp_min(1.0)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_fingerprint(tensor: torch.Tensor, limit: int = 4096) -> str:
    flat = tensor.detach().to(torch.float32).reshape(-1).cpu().numpy()
    return _sha256_bytes(flat[:limit].tobytes())


def _git_sha(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("torch", "transformers", "datasets", "speculators"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[name] = "unavailable"
    return versions


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _copy_config_only(source_dir: Path, target_dir: Path) -> Path:
    source = source_dir / "config.json"
    if not source.is_file():
        raise FileNotFoundError(f"Missing config.json under {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=False)
    target = target_dir / "config.json"
    shutil.copy2(source, target)
    weight_globs = ("*.safetensors", "*.bin", "*.safetensors.index.json")
    leaked = [p for pattern in weight_globs for p in target_dir.glob(pattern)]
    if leaked:
        raise RuntimeError(f"Config-only init directory contains weights: {leaked}")
    return target


def _validate_config(config, verifier_path: Path) -> None:
    if config.speculators_model_type not in ("dflash", "dspark"):
        raise ValueError(
            f"Expected dflash/dspark config, got {config.speculators_model_type}"
        )
    if config.block_size != 8:
        raise ValueError(f"Expected block_size=8, got {config.block_size}")
    if config.draft_vocab_size != config.transformer_layer_config.vocab_size:
        raise ValueError(
            "Expected the draft vocab to equal the verifier vocab "
            f"({config.transformer_layer_config.vocab_size}), got "
            f"{config.draft_vocab_size}"
        )
    if list(config.aux_hidden_state_layer_ids or []) != [2, 18, 33]:
        raise ValueError(
            "Expected target layers [2, 18, 33], got "
            f"{config.aux_hidden_state_layer_ids}"
        )
    if config.transformer_layer_config.num_hidden_layers != 5:
        raise ValueError(
            "Expected 5 draft layers, got "
            f"{config.transformer_layer_config.num_hidden_layers}"
        )
    saved_verifier = Path(config.speculators_config.verifier.name_or_path or "")
    if not saved_verifier.exists():
        config.speculators_config.verifier.name_or_path = str(verifier_path)


def build_draft(
    config_dir: Path,
    data_path: Path,
    verifier_path: Path,
    init_draft_path: Path | None = None,
    *,
    markov_rank: int = 256,
    markov_head_type: str = "vanilla",
    enable_confidence_head: bool = True,
    confidence_head_with_markov: bool = True,
):
    base_config = DFlashSpeculatorConfig.from_pretrained(config_dir)
    _validate_config(base_config, verifier_path)

    t2d_path = data_path / "t2d.npy"
    d2t_path = data_path / "d2t.npy"
    if t2d_path.exists() and d2t_path.exists():
        t2d = torch.from_numpy(np.load(t2d_path))
        d2t = torch.from_numpy(np.load(d2t_path))
    else:
        t2d = d2t = None

    if init_draft_path is not None:
        model = DSparkDraftModel.from_pretrained(
            str(init_draft_path),
            verifier=str(verifier_path),
            t2d=t2d,
            d2t=d2t,
            local_files_only=True,
        )
        return model, model.config, True

    model = DSparkDraftModel.from_training_args(
        base_config.transformer_layer_config,
        t2d=t2d,
        d2t=d2t,
        verifier_name_or_path=str(verifier_path),
        draft_vocab_size=base_config.draft_vocab_size,
        block_size=base_config.block_size,
        target_layer_ids=list(base_config.aux_hidden_state_layer_ids or []),
        sample_from_anchor=base_config.sample_from_anchor,
        mask_token_id=base_config.mask_token_id,
        sliding_window_non_causal=base_config.sliding_window_non_causal,
        markov_rank=markov_rank,
        markov_head_type=markov_head_type,
        enable_confidence_head=enable_confidence_head,
        confidence_head_with_markov=confidence_head_with_markov,
    )
    return model, model.config, False


def init_audit(model, config_dir: Path, config_sha: str, seed: int,
               draft_reference: Path, loaded_from_checkpoint: bool = False) -> dict:
    audit = {
        "seed": seed,
        "init_config_dir": str(config_dir),
        "init_config_sha256": config_sha,
        "weights_loaded_from_draft_checkpoint": loaded_from_checkpoint,
        "trainable_init": (
            "loaded from draft checkpoint"
            if loaded_from_checkpoint
            else "random (DFlash post_init)"
        ),
        "frozen_init": "verifier embed_tokens/lm_head/verifier_lm_head/"
        "verifier_norm",
        "draft_reference_checkpoint": str(draft_reference),
        "random_samples": {},
        "reference_samples": {},
        "weights_differ_from_reference": {},
    }
    reference_weights = {}
    reference_file = draft_reference / "model.safetensors"
    if reference_file.is_file():
        from safetensors import safe_open  # noqa: PLC0415

        with safe_open(str(reference_file), framework="pt", device="cpu") as handle:
            for key in (
                "layers.0.self_attn.q_proj.weight",
                "layers.4.mlp.down_proj.weight",
                "fc.weight",
            ):
                if key in handle.keys():
                    reference_weights[key] = handle.get_tensor(key)

    for key, tensor in model.state_dict().items():
        if key in reference_weights:
            random_sha = _tensor_fingerprint(tensor)
            reference_sha = _tensor_fingerprint(reference_weights[key])
            audit["random_samples"][key] = random_sha
            audit["reference_samples"][key] = reference_sha
            audit["weights_differ_from_reference"][key] = random_sha != reference_sha
    compared = audit["weights_differ_from_reference"]
    audit["all_compared_weights_differ"] = bool(compared) and all(compared.values())
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--init-config-source",
        default=str(PROJECT_ROOT / "qwen3_4b_dflash_300k" / "checkpoints" / "3"),
        help="Directory whose config.json defines the architecture; weights are "
        "never read from it.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--epochs-per-run", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-config-change", action="store_true")

    parser.add_argument("--data-path", default=str(PROJECT_ROOT / "qwen3_4b_dflash_300k" / "prepared"))
    parser.add_argument(
        "--hidden-states-path",
        default=str(PROJECT_ROOT / "qwen3_4b_dflash_300k" / "hidden_states"),
    )
    parser.add_argument("--verifier-path", default=str(PROJECT_ROOT / "Qwen3-4B-Instruct"))
    parser.add_argument("--train-data-ratio", type=float, default=0.999)
    parser.add_argument("--total-seq-len", type=int, default=8192)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--prefetch-factor", type=int, default=4)

    parser.add_argument("--max-anchors", type=int, default=3072)
    parser.add_argument("--verify-batch", type=int, default=64)
    parser.add_argument("--chunk-blocks", type=int, default=256)
    parser.add_argument("--markov-rank", type=int, default=256)
    parser.add_argument(
        "--markov-head-type",
        choices=("vanilla", "gated", "rnn"),
        default="vanilla",
    )
    parser.add_argument(
        "--enable-confidence-head",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--confidence-head-with-markov",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--confidence-head-alpha", type=float, default=1.0)
    parser.add_argument("--verifier-attn-impl", default="sdpa")
    parser.add_argument(
        "--verifier-label-backend",
        choices=("per_anchor", "tree"),
        default="per_anchor",
        help="per_anchor: one verifier forward per anchor (validated default). "
        "tree: share document prefixes with a block-sparse tree mask; labels "
        "use the draft-generated prefix for every slot.",
    )
    parser.add_argument(
        "--online-loss",
        choices=("soft_eal", "log_eal"),
        default="soft_eal",
        help="soft_eal: current -mean(EAL). log_eal: -log(mean(EAL)+eps).",
    )
    parser.add_argument("--log-eal-eps", type=float, default=1e-6)
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=0.0,
        help="Coefficient for the prefix-weighted forward KL term. "
        "Requires --verifier-label-backend tree.",
    )
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument(
        "--log-position-diagnostics",
        action="store_true",
        help="Log per-position valid/correct/prefix-ok/wasted counts.",
    )
    parser.add_argument(
        "--init-draft-path",
        default=None,
        help="Optional draft checkpoint to initialize from (post-training). "
        "Omit for from-scratch random initialization.",
    )

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--muon-lr", type=float, default=None)
    parser.add_argument("--muon-momentum", type=float, default=0.95)
    parser.add_argument("--muon-weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument(
        "--muon-adjust-lr-fn", choices=("original", "match_rms_adamw"),
        default="match_rms_adamw",
    )
    parser.add_argument("--scheduler-warmup-steps", type=int, default=None)
    parser.add_argument("--scheduler-cosine-cycles", type=float, default=0.5)
    parser.add_argument("--checkpoint-freq", type=float, default=0.01)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Build the random draft + loader, write audits, then exit.",
    )
    return parser.parse_args()


def make_train_loader(args: argparse.Namespace, config, verifier_path: Path):
    data_path = Path(args.data_path)
    transfer = FileTransfer(Path(args.hidden_states_path))
    train_loader, _ = create_train_val_loaders(
        data_path=str(data_path),
        total_seq_len=args.total_seq_len,
        hidden_states_dtype=torch.bfloat16,
        noise_std=args.noise_std,
        legacy_data=False,
        transfer=transfer,
        vllm_endpoint="http://localhost:8000/v1",
        on_missing="raise",
        on_generate="cache",
        verifier_name_or_path=str(verifier_path),
        request_timeout=3600,
        max_retries=3,
        hidden_size=config.transformer_layer_config.hidden_size,
        num_target_layers=len(config.aux_hidden_state_layer_ids),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        preprocess=None,
        train_data_ratio=args.train_data_ratio,
    )
    return train_loader


def _context_starts(document_ids: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(document_ids.numel(), device=document_ids.device)
    is_start = torch.ones_like(positions, dtype=torch.bool)
    is_start[1:] = document_ids[1:] != document_ids[:-1]
    starts = torch.where(is_start, positions, torch.zeros_like(positions))
    return torch.cummax(starts, dim=0).values


def online_verifier_labels(
    verifier,
    input_ids: torch.Tensor,
    document_ids: torch.Tensor,
    anchors: torch.Tensor,
    proposals: torch.Tensor,
    verify_batch: int,
    device: torch.device,
) -> torch.Tensor:
    """Return greedy verifier labels for ``true prefix + draft proposal``.

    Every anchor is an independent verifier sequence.  Contexts are sorted by
    length only to reduce padding; labels are returned in the input order.
    """
    starts = _context_starts(document_ids)
    selected_anchor = anchors.tolist()
    selected_start = starts[anchors].tolist()
    ids = input_ids.tolist()
    contexts = []
    for start, anchor in zip(selected_start, selected_anchor, strict=True):
        if not 0 <= start <= anchor < len(ids):
            raise RuntimeError(f"Invalid context [{start}, {anchor}]")
        contexts.append(ids[start : anchor + 1])
    proposal_list = proposals.tolist()
    num_slots = len(proposal_list[0]) if proposal_list else 0

    order = sorted(range(len(contexts)), key=lambda idx: len(contexts[idx]))
    sorted_contexts = [contexts[idx] for idx in order]
    sorted_proposals = [proposal_list[idx] for idx in order]
    labels = torch.zeros(
        (len(contexts), num_slots), dtype=torch.long, device=device
    )

    for batch_start in range(0, len(sorted_contexts), verify_batch):
        batch_end = min(batch_start + verify_batch, len(sorted_contexts))
        batch_contexts = sorted_contexts[batch_start:batch_end]
        batch_proposals = sorted_proposals[batch_start:batch_end]
        lengths = [len(ctx) for ctx in batch_contexts]
        width = max(lengths) + num_slots
        tokens = torch.zeros((len(lengths), width), dtype=torch.long)
        attention = torch.zeros_like(tokens)
        positions = torch.zeros_like(tokens)
        for row, (ctx, proposal) in enumerate(
            zip(batch_contexts, batch_proposals, strict=True)
        ):
            sequence = ctx + proposal
            length = len(sequence)
            tokens[row, :length] = torch.tensor(sequence, dtype=torch.long)
            attention[row, :length] = 1
            positions[row, :length] = torch.arange(length, dtype=torch.long)
        tokens = tokens.to(device, non_blocking=True)
        attention = attention.to(device, non_blocking=True)
        positions = positions.to(device, non_blocking=True)

        slot_indices = torch.tensor(
            [[length - 1 + slot for slot in range(num_slots)] for length in lengths],
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            hidden = verifier.model(
                input_ids=tokens,
                attention_mask=attention,
                position_ids=positions,
                use_cache=False,
            ).last_hidden_state
            gathered = hidden.gather(
                1, slot_indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            )
            predicted = verifier.lm_head(gathered).argmax(dim=-1)
        destination = torch.tensor(
            order[batch_start:batch_end], dtype=torch.long, device=device
        )
        labels[destination] = predicted
    return labels


def offline_verifier_labels_chunked(
    model,
    verifier_last_hidden_states: torch.Tensor,
    blocks: torch.Tensor,
    chunk_blocks: int = 256,
) -> torch.Tensor:
    """Greedy offline labels from the original target prefix.

    Used only for position diagnostics: labels are computed at the original
    sequence positions and do not depend on draft proposals.
    """
    total_seq_len = verifier_last_hidden_states.shape[1]
    slot_positions = blocks[:, 1:]
    if getattr(model.config, "sample_from_anchor", False):
        target_positions = slot_positions
    else:
        target_positions = (slot_positions - 1) % total_seq_len
    labels = torch.empty(
        slot_positions.shape, dtype=torch.long, device=blocks.device
    )
    weight = getattr(model.verifier_lm_head, "weight", None)
    with torch.no_grad():
        for start in range(0, blocks.shape[0], chunk_blocks):
            end = min(start + chunk_blocks, blocks.shape[0])
            hidden = verifier_last_hidden_states[:, target_positions[start:end]]
            if weight is not None:
                hidden = hidden.to(dtype=weight.dtype)
            logits = model.verifier_lm_head(model.verifier_norm(hidden))
            labels[start:end] = logits.argmax(dim=-1)[0]
    return labels


class OnlineTrainer(Trainer):
    def __init__(
        self,
        model,
        config: TrainerConfig,
        train_loader,
        verifier,
        *,
        seed: int,
        max_anchors: int,
        verify_batch: int,
        chunk_blocks: int,
        grad_clip_norm: float,
        metrics_path: Path,
        log_every: int,
        verifier_label_backend: str,
        online_loss: str,
        log_eal_eps: float,
        kl_weight: float = 0.0,
        kl_temperature: float = 1.0,
        confidence_head_alpha: float = 0.0,
        log_position_diagnostics: bool = False,
    ):
        if kl_weight > 0 and verifier_label_backend != "tree":
            raise ValueError(
                "kl_weight > 0 requires --verifier-label-backend tree"
            )
        if kl_temperature <= 0:
            raise ValueError("kl_temperature must be positive")
        self.verifier = verifier
        self.seed = seed
        self.max_anchors = max_anchors
        self.verify_batch = verify_batch
        self.chunk_blocks = chunk_blocks
        self.grad_clip_norm = grad_clip_norm
        self.metrics_path = metrics_path
        self.log_every = log_every
        self.verifier_label_backend = verifier_label_backend
        self.online_loss = online_loss
        self.log_eal_eps = log_eal_eps
        self.kl_weight = kl_weight
        self.kl_temperature = kl_temperature
        self.confidence_head_alpha = confidence_head_alpha
        self.log_position_diagnostics = log_position_diagnostics
        self._tree_compile_logged = False
        self.process_start = time.perf_counter()
        super().__init__(model, config, train_loader, None)

    def setup_model(self):
        """Keep one replicated model per rank; gradients are averaged manually."""
        SpeculatorModel.verify_training_compatible(self.model)
        self.model.to(self.local_rank)
        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            self.checkpointer.load_model_state_dict(self.model)
        if self.is_distributed:
            for param in self.model.parameters():
                dist.broadcast(param.data, src=0)
            dist.barrier()

    def _online_forward(self, raw_model, batch):
        device = batch["input_ids"].device

        t0 = time.perf_counter()
        with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
            hidden, logits, _targets, aligned_mask, block_indices = (
                raw_model.online_backbone_forward(
                    hidden_states=batch["hidden_states"],
                    input_ids=batch["input_ids"],
                    loss_mask=batch["loss_mask"],
                    verifier_last_hidden_states=batch[
                        "verifier_last_hidden_states"
                    ],
                    document_ids=batch["document_ids"],
                    position_ids=batch.get("position_ids"),
                    max_anchors=self.max_anchors,
                )
            )
        draft_forward_s = time.perf_counter() - t0

        block_size = raw_model.block_size
        num_blocks = block_indices.numel() // block_size
        blocks = block_indices.reshape(num_blocks, block_size)
        anchors = blocks[:, 0]
        hidden_blocks = hidden.reshape(num_blocks, block_size, -1)
        unary = logits.reshape(num_blocks, block_size, -1)
        block_tokens = batch["input_ids"][0][blocks]
        # DSpark: walk the Markov chain with the previously selected token,
        # matching serving and keeping the EAL/KL logits on the draft prefix.
        prev = block_tokens[:, 0]
        corrected = []
        prev_embs = []
        for slot in range(1, block_size):
            prev_emb = raw_model.markov_head.prev_embeddings(prev.unsqueeze(1))
            bias = raw_model.markov_head.block_bias(
                prev_token_ids=prev.unsqueeze(1),
                hidden_states=hidden_blocks[:, slot : slot + 1],
                prev_emb=prev_emb,
            )[:, 0]
            scores_slot = unary[:, slot] + bias.to(unary.dtype)
            corrected.append(scores_slot)
            prev_embs.append(prev_emb[:, 0])
            prev = scores_slot.detach().argmax(-1)
        scores = torch.stack(corrected, dim=1)
        proposal = scores.detach().argmax(-1)
        prev_emb_stack = torch.stack(prev_embs, dim=1)
        # Drop the per-slot graph tensors before the chunked loss; keeping the
        # list alongside the stacked copy costs another full logits tensor.
        del corrected, unary, logits

        docs = batch["document_ids"][0]
        slot_mask = aligned_mask.reshape(num_blocks, block_size)[:, 1:].bool()
        valid = slot_mask & docs[blocks[:, 1:]].eq(docs[anchors, None])
        valid = valid.long().cumprod(-1).bool()
        selected = valid.any(-1).nonzero().flatten()
        if selected.numel() == 0:
            raise RuntimeError("No valid draft prefix in batch")

        t1 = time.perf_counter()
        labels = torch.zeros_like(proposal)
        target_branch_hidden = None
        if self.verifier_label_backend == "tree":
            if not self._tree_compile_logged:
                LOG.info(
                    "Tree label backend first call: compiling flex attention / "
                    "block-mask kernels (one-time, may take minutes)"
                )
                self._tree_compile_logged = True
            if self.kl_weight > 0:
                labels_tree, target_branch_hidden = online_verifier_labels_tree(
                    self.verifier,
                    batch["input_ids"],
                    batch["document_ids"],
                    anchors[selected],
                    proposal[selected],
                    max_branches=self.max_anchors,
                    use_flex=True,
                    return_hidden=True,
                )
                labels[selected] = labels_tree
            else:
                labels[selected] = online_verifier_labels_tree(
                    self.verifier,
                    batch["input_ids"],
                    batch["document_ids"],
                    anchors[selected],
                    proposal[selected],
                    max_branches=self.max_anchors,
                    use_flex=True,
                )
        else:
            labels[selected] = online_verifier_labels(
                self.verifier,
                batch["input_ids"][0],
                docs,
                anchors[selected],
                proposal[selected],
                self.verify_batch,
                device,
            )
        verifier_s = time.perf_counter() - t1

        t2 = time.perf_counter()
        if selected.numel() == num_blocks:
            scores_sel, labels_sel, valid_sel = scores, labels, valid
        else:
            scores_sel = scores[selected]
            labels_sel = labels[selected]
            valid_sel = valid[selected]
        loss_eal = soft_eal_loss_chunked(
            scores_sel,
            labels_sel,
            valid_sel,
            chunk_blocks=self.chunk_blocks,
            log_eal=self.online_loss == "log_eal",
            eps=self.log_eal_eps,
        )
        loss_kl = None
        loss = loss_eal
        if self.kl_weight > 0:
            assert target_branch_hidden is not None  # noqa: S101
            kl_weights = prefix_weights_logspace(
                scores_sel,
                labels_sel,
                valid_sel,
                chunk_blocks=self.chunk_blocks,
            )
            loss_kl = forward_kl_chunked(
                scores_sel,
                target_branch_hidden,
                raw_model.verifier_lm_head,
                valid_mask=valid_sel,
                weights=kl_weights,
                chunk_blocks=self.chunk_blocks,
                temperature=self.kl_temperature,
            )
            loss = loss_eal + self.kl_weight * loss_kl
        loss_conf = None
        if self.confidence_head_alpha > 0 and raw_model.confidence_head is not None:
            conf_features = torch.cat(
                [
                    hidden_blocks[selected][:, 1:],
                    prev_emb_stack[selected].to(hidden_blocks.dtype),
                ],
                dim=-1,
            )
            conf_head = raw_model.confidence_head
            conf_weight = next(conf_head.parameters(), None)
            if conf_weight is not None:
                conf_features = conf_features.to(conf_weight.dtype)
            confidence_logits = conf_head(conf_features)
            # Greedy acceptance target: the tree label is the verifier token
            # accepted at this draft prefix, so this is the per-slot indicator.
            conf_target = (
                proposal[selected].eq(labels_sel) & valid_sel
            ).to(torch.float32)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                confidence_logits.float(), conf_target, reduction="none"
            )
            mask = valid_sel.to(bce.dtype)
            loss_conf = (bce * mask).sum() / mask.sum().clamp_min(1.0)
            loss = loss + self.confidence_head_alpha * loss_conf
        loss_s = time.perf_counter() - t2

        proposal_sel = proposal[selected]
        labels_sel = labels[selected]
        valid_sel = valid[selected]
        accepted = (proposal_sel.eq(labels_sel) & valid_sel).long().cumprod(-1).sum(-1)
        greedy_eal = 1.0 + float(accepted.float().mean())
        partial_blocks = int((valid_sel.sum(-1) < block_size - 1).sum())
        detail = {
            "online_loss": self.online_loss,
            "loss_eal": float(loss_eal.detach()),
            "loss_kl": 0.0 if loss_kl is None else float(loss_kl.detach()),
            "loss_conf": 0.0 if loss_conf is None else float(loss_conf.detach()),
            "kl_weight": self.kl_weight,
            "blocks": int(selected.numel()),
            "partial_blocks": partial_blocks,
            "greedy_eal": greedy_eal,
            "draft_forward_s": draft_forward_s,
            "verifier_s": verifier_s,
            "loss_s": loss_s,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30,
        }
        if self.log_position_diagnostics:
            target_tokens = batch["input_ids"][0][blocks[selected, 1:]]
            correct = proposal_sel.eq(labels_sel) & valid_sel
            prefix_ok = torch.ones_like(correct)
            prefix_ok[:, 1:] = correct[:, :-1].long().cumprod(-1).bool()
            prefix_ok = prefix_ok & valid_sel
            wasted = correct & ~prefix_ok
            true_correct = proposal_sel.eq(target_tokens) & valid_sel
            true_wasted = true_correct & ~prefix_ok
            offline_labels = offline_verifier_labels_chunked(
                raw_model,
                batch["verifier_last_hidden_states"],
                blocks[selected],
                self.chunk_blocks,
            )
            off_correct = proposal_sel.eq(offline_labels) & valid_sel
            off_prefix_ok = torch.ones_like(off_correct)
            off_prefix_ok[:, 1:] = off_correct[:, :-1].long().cumprod(-1).bool()
            off_prefix_ok = off_prefix_ok & valid_sel
            off_wasted = off_correct & ~off_prefix_ok
            detail.update(
                {
                    "pos_valid": valid_sel.sum(0).tolist(),
                    "pos_correct": correct.sum(0).tolist(),
                    "pos_prefix_ok": prefix_ok.sum(0).tolist(),
                    "pos_wasted": wasted.sum(0).tolist(),
                    "pos_true_correct": true_correct.sum(0).tolist(),
                    "pos_true_wasted": true_wasted.sum(0).tolist(),
                    "pos_off_correct": off_correct.sum(0).tolist(),
                    "pos_off_prefix_ok": off_prefix_ok.sum(0).tolist(),
                    "pos_off_wasted": off_wasted.sum(0).tolist(),
                }
            )
        return loss, detail

    def train_epoch(self, epoch: int):  # noqa: C901
        if (
            self.config.max_steps is not None
            and self.global_step >= self.config.max_steps
        ):
            LOG.warning("max_steps already reached; skipping epoch %s", epoch)
            return
        self.model.train()
        if hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)
        num_steps = len(self.train_loader)
        skip_steps = self._prepare_resume_skip(epoch)
        # Deterministic worker seeding for this epoch.  The per-step anchor seed
        # below is drawn after workers have been seeded, so it does not perturb
        # the noise stream.
        torch.manual_seed(self.seed + epoch * 7919)
        step_interval = (
            max(1, round(num_steps * self.config.checkpoint_freq))
            if self.config.checkpoint_freq < 1
            else None
        )
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        torch.cuda.reset_peak_memory_stats()
        epoch_start = time.perf_counter()

        for local_step_rel, batch in enumerate(self.train_loader, 1):
            local_step = local_step_rel + skip_steps
            global_step = self.global_step
            # Deterministic anchor sampling across process restarts.  The
            # distribution (uniform over valid positions) is unchanged.
            torch.manual_seed(self.seed + epoch * 1_000_003 + local_step)

            gpu_batch = {
                key: value.to(self.local_rank, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }
            step_start = time.perf_counter()
            loss, detail = self._online_forward(raw_model, gpu_batch)
            self._optimizers_zero_grad()
            loss.backward()
            if self.is_distributed:
                params = [
                    p for p in raw_model.parameters() if p.grad is not None
                ]
                grads = [p.grad for p in params]
                flat = _flatten_dense_tensors(grads)
                dist.all_reduce(flat, op=dist.ReduceOp.SUM)
                flat.div_(dist.get_world_size())
                for param, grad in zip(
                    params,
                    _unflatten_dense_tensors(flat, grads),
                    strict=True,
                ):
                    param.grad = grad
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in raw_model.parameters() if p.requires_grad],
                self.grad_clip_norm,
                error_if_nonfinite=True,
            )
            lrs = {
                type(opt).__name__: opt.param_groups[0]["lr"]
                for opt in self.optimizers
            }
            self._optimizers_step()
            self._schedulers_step()
            self.global_step += 1

            row = {
                "epoch": epoch,
                "rank": self.rank,
                "local_step": local_step,
                "global_step": global_step,
                "loss": float(loss.detach()),
                "grad_norm": float(grad_norm),
                "lr": lrs,
                "step_s": time.perf_counter() - step_start,
                "elapsed_epoch_s": time.perf_counter() - epoch_start,
                "elapsed_process_s": time.perf_counter() - self.process_start,
                **detail,
            }
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            if local_step_rel % self.log_every == 0 or local_step_rel == 1:
                print(json.dumps(row), flush=True)

            if (
                self.config.max_steps is not None
                and self.global_step >= self.config.max_steps
            ):
                LOG.warning(
                    "Reached max_steps=%s at global_step=%s",
                    self.config.max_steps,
                    self.global_step,
                )
                break

            if (
                step_interval is not None
                and not self.config.save_best
                and local_step % step_interval == 0
                and num_steps - local_step >= step_interval * MIN_STEP_PCT
            ):
                self.maybe_save_checkpoint(epoch, local_step=local_step)

        LOG.info(
            "Epoch %s finished: steps=%s global_step=%s elapsed=%.1fs",
            epoch,
            num_steps,
            self.global_step,
            time.perf_counter() - epoch_start,
        )


def main() -> int:
    args = parse_args()
    if args.max_anchors <= 0 or args.verify_batch <= 0 or args.chunk_blocks <= 0:
        raise ValueError("max_anchors, verify_batch and chunk_blocks must be positive")
    if not 0 < args.train_data_ratio < 1:
        raise ValueError("train_data_ratio must be in (0, 1)")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    maybe_setup_distributed()
    rank = get_rank()
    world_size = get_world_size()
    LOG.info("Distributed: rank=%s world_size=%s", rank, world_size)
    run_root = Path(args.run_root).resolve()
    checkpoints_dir = run_root / "checkpoints"
    init_config_dir = run_root / "init_config"
    metrics_path = run_root / (
        "metrics.jsonl" if rank == 0 else f"metrics_rank{rank}.jsonl"
    )
    signature_path = run_root / "run_signature.json"

    if rank == 0:
        has_checkpoints = checkpoints_dir.is_dir() and any(
            child.is_dir() and child.name.isdigit()
            for child in checkpoints_dir.iterdir()
        )
        if has_checkpoints and not args.resume:
            raise SystemExit(
                f"{run_root} already has checkpoints; pass --resume or use a new --run-root"
            )
        if args.resume and not has_checkpoints:
            raise SystemExit(
                f"--resume requested but no epoch checkpoint under {checkpoints_dir}"
            )
        if not args.resume:
            allowed = {"pipeline.log", "pipeline_state.json", "driver.out"}
            existing = (
                {child.name for child in run_root.iterdir()}
                if run_root.exists()
                else set()
            )
            unexpected = existing - allowed
            if unexpected:
                raise SystemExit(
                    f"{run_root} exists and is not empty ({sorted(unexpected)}); "
                    "use a new --run-root"
                )
            run_root.mkdir(parents=True, exist_ok=True)
        elif not run_root.exists():
            raise SystemExit(f"--resume requested but {run_root} does not exist")
    if is_distributed():
        dist.barrier()

    data_path = Path(args.data_path).resolve()
    hidden_states_path = Path(args.hidden_states_path).resolve()
    verifier_path = Path(args.verifier_path).resolve()
    init_source = Path(args.init_config_source).resolve()
    init_draft_path = (
        Path(args.init_draft_path).resolve() if args.init_draft_path else None
    )
    if init_draft_path is not None and not init_draft_path.exists():
        raise FileNotFoundError(init_draft_path)
    for path in (data_path, hidden_states_path, verifier_path):
        if not path.exists():
            raise FileNotFoundError(path)

    if rank == 0 and not args.resume:
        _copy_config_only(init_source, init_config_dir)
    if is_distributed():
        dist.barrier()
    config_file = init_config_dir / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(config_file)
    config_sha = _sha256_file(config_file)

    _set_seed(args.seed)
    model, config, loaded_from_checkpoint = build_draft(
        init_config_dir,
        data_path,
        verifier_path,
        init_draft_path,
        markov_rank=args.markov_rank,
        markov_head_type=args.markov_head_type,
        enable_confidence_head=args.enable_confidence_head,
        confidence_head_with_markov=args.confidence_head_with_markov,
    )
    if loaded_from_checkpoint:
        LOG.info("Draft initialized from checkpoint: %s", init_draft_path)
    else:
        LOG.info(
            "Draft initialized from config only (no draft checkpoint weights): %s",
            init_config_dir,
        )
    if not args.resume and rank == 0:
        audit = init_audit(
            model,
            init_config_dir,
            config_sha,
            args.seed,
            init_draft_path or init_source,
            loaded_from_checkpoint=loaded_from_checkpoint,
        )
        _write_json(run_root / "init_audit.json", audit)
        LOG.info(
            "Init audit: loaded_from_checkpoint=%s compared=%s all_differ=%s",
            loaded_from_checkpoint,
            len(audit["weights_differ_from_reference"]),
            audit.get("all_compared_weights_differ"),
        )

    train_loader = make_train_loader(args, config, verifier_path)
    sampler = train_loader.batch_sampler
    epoch_steps = [len(sampler._generate_batches(e)) for e in range(args.epochs)]
    total_steps = sum(epoch_steps)
    warmup_steps = (
        args.scheduler_warmup_steps
        if args.scheduler_warmup_steps is not None
        else total_steps // 100
    )
    signature = {
        "seed": args.seed,
        "init_config_sha256": config_sha,
        "max_anchors": args.max_anchors,
        "verify_batch": args.verify_batch,
        "chunk_blocks": args.chunk_blocks,
        "verifier_label_backend": args.verifier_label_backend,
        "online_loss": args.online_loss,
        "log_eal_eps": args.log_eal_eps,
        "init_draft_path": str(init_draft_path) if init_draft_path else None,
        "random_init": not loaded_from_checkpoint,
        "train_data_ratio": args.train_data_ratio,
        "total_seq_len": args.total_seq_len,
        "noise_std": args.noise_std,
        "epochs": args.epochs,
        "lr": args.lr,
        "optimizer": "muon+adamw",
        "optimizer_hparams": {
            "weight_decay": args.weight_decay,
            "muon_lr": args.muon_lr if args.muon_lr is not None else 10 * args.lr,
            "muon_momentum": args.muon_momentum,
            "muon_weight_decay": args.muon_weight_decay,
            "muon_ns_steps": args.muon_ns_steps,
            "muon_adjust_lr_fn": args.muon_adjust_lr_fn,
        },
        "scheduler": {
            "type": "cosine",
            "cycles": args.scheduler_cosine_cycles,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
        },
        "steps_per_epoch": epoch_steps,
        "total_steps": total_steps,
        "data_path": str(data_path),
        "hidden_states_path": str(hidden_states_path),
        "world_size": world_size,
        "distributed": is_distributed(),
    }
    if rank == 0:
        if args.resume:
            saved = json.loads(signature_path.read_text())
            differences = {
                key: (saved.get(key), value)
                for key, value in signature.items()
                if saved.get(key) != value
            }
            if differences and not args.allow_config_change:
                raise SystemExit(
                    "Resume config differs from the original run: "
                    + json.dumps(differences, ensure_ascii=False)
                    + "; pass --allow-config-change to override"
                )
        else:
            _write_json(signature_path, signature)

        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        command_text = (
            f"# Timestamp: {datetime.now(timezone.utc).isoformat()}\n"
            f"# Git SHA (speculators-src): {_git_sha(PROJECT_ROOT / 'speculators-src')}\n"
            f"# Versions: {json.dumps(_package_versions())}\n"
            f"# Seed: {args.seed}\n"
            f"# Distributed: world_size={world_size}, launcher=torchrun\n"
            f"# Init config: {init_config_dir} (sha256={config_sha})\n"
            f"# Draft init: "
            f"{init_draft_path if loaded_from_checkpoint else 'config-only random'}\n"
            f"# Draft checkpoint weights loaded: {loaded_from_checkpoint}\n"
            f"# Online loss: {args.online_loss} (eps={args.log_eal_eps})\n"
            f"# Resolved steps/epoch (per rank): {epoch_steps}; total steps: "
            f"{total_steps}; warmup: {warmup_steps}\n"
            f"# Resolved run config: {json.dumps(vars(args), ensure_ascii=False)}\n"
            + subprocess.list2cmdline([sys.executable, *sys.argv])
            + "\n"
        )
        (checkpoints_dir / "train_command.txt").write_text(command_text)
        (run_root / "train_command.txt").write_text(command_text)
        _write_json(run_root / "resolved_run_config.json", {
            **vars(args),
            "steps_per_epoch": epoch_steps,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "world_size": world_size,
            "distributed": is_distributed(),
        })
    if is_distributed():
        dist.barrier()

    if args.init_only:
        if rank == 0:
            LOG.info(
                "Init-only check complete: steps/epoch=%s total_steps=%s warmup=%s",
                epoch_steps,
                total_steps,
                warmup_steps,
            )
        maybe_destroy_distributed()
        return 0

    LOG.info("Loading frozen verifier: %s", verifier_path)
    verifier = (
        AutoModelForCausalLM.from_pretrained(
            str(verifier_path),
            dtype=torch.bfloat16,
            local_files_only=True,
            attn_implementation=args.verifier_attn_impl,
        )
        .eval()
        .requires_grad_(False)
        .to(get_local_rank())
    )

    muon_lr = args.muon_lr if args.muon_lr is not None else 10 * args.lr
    trainer_config = TrainerConfig(
        lr=args.lr,
        num_epochs=args.epochs,
        save_path=str(checkpoints_dir),
        resume_from_checkpoint=True,
        optimizer="muon",
        weight_decay=args.weight_decay,
        muon_lr=muon_lr,
        muon_momentum=args.muon_momentum,
        muon_weight_decay=args.muon_weight_decay,
        muon_ns_steps=args.muon_ns_steps,
        muon_adjust_lr_fn=args.muon_adjust_lr_fn,
        scheduler_type="cosine",
        scheduler_warmup_steps=args.scheduler_warmup_steps,
        scheduler_total_steps=total_steps,
        scheduler_num_cosine_cycles=args.scheduler_cosine_cycles,
        checkpoint_freq=args.checkpoint_freq,
        save_best=False,
        hidden_states_dtype=torch.bfloat16,
        log_freq=1,
        fsdp_shard=False,
        max_steps=args.max_steps,
    )

    trainer = OnlineTrainer(
        model,
        trainer_config,
        train_loader,
        verifier,
        seed=args.seed,
        max_anchors=args.max_anchors,
        verify_batch=args.verify_batch,
        chunk_blocks=args.chunk_blocks,
        grad_clip_norm=args.grad_clip_norm,
        metrics_path=metrics_path,
        log_every=args.log_every,
        verifier_label_backend=args.verifier_label_backend,
        online_loss=args.online_loss,
        log_eal_eps=args.log_eal_eps,
        kl_weight=args.kl_weight,
        kl_temperature=args.kl_temperature,
        confidence_head_alpha=args.confidence_head_alpha,
        log_position_diagnostics=args.log_position_diagnostics,
    )
    run_epochs = min(args.epochs_per_run, args.epochs - trainer.current_epoch)
    if run_epochs <= 0:
        LOG.info("All %s epochs already completed", args.epochs)
        maybe_destroy_distributed()
        return 0
    trainer.config = trainer.config._replace(
        num_epochs=trainer.current_epoch + run_epochs
    )
    LOG.info(
        "Training epochs %s..%s (resume=%s, current_epoch=%s, global_step=%s)",
        trainer.current_epoch,
        trainer.current_epoch + run_epochs - 1,
        args.resume,
        trainer.current_epoch,
        trainer.global_step,
    )
    trainer.run_training()
    LOG.info("Training run complete; latest checkpoint dir: %s", checkpoints_dir)

    del trainer, model, verifier
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    maybe_destroy_distributed()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
