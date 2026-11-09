"""Prefix-sharing (tree) verifier label generation for online soft-EAL.

The current per-anchor implementation reruns the verifier over the full
document prefix for every anchor.  A packed batch contains only a handful of
documents, so those prefixes are heavily shared.  This module processes each
packed document once and appends one causal branch per anchor:

    base:     true packed document tokens (causal within each document)
    branch b: [anchor_token, proposal_0, ..., proposal_6]
              may attend to base tokens strictly before its anchor and to
              earlier tokens of its own branch only.

Labels 1..7 are read from branch slots 0..6 (anchor + first six proposals);
proposal_6 is fed only to match the existing per-anchor 7-token input and its
hidden state is not used.

The mask is a block-sparse tree mask executed with torch flex attention.  The
sequence length is padded to ``base_len + max_branches * block_size`` so the
compiled flex kernel keeps a fixed shape across packed batches.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

LABEL_SLOTS = 7
PROPOSAL_TOKENS = 7  # anchor + 7 proposals; labels use the first 7 hidden slots

_COMPILED_CREATE_BLOCK_MASK = None
_COMPILED_FLEX_ATTENTION = None


def _context_starts(document_ids: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(document_ids.numel(), device=document_ids.device)
    is_start = torch.ones_like(positions, dtype=torch.bool)
    is_start[1:] = document_ids[1:] != document_ids[:-1]
    starts = torch.where(is_start, positions, torch.zeros_like(positions))
    return torch.cummax(starts, dim=0).values


def _get_create_block_mask(device: torch.device):
    global _COMPILED_CREATE_BLOCK_MASK
    if device.type != "cuda":
        return create_block_mask
    if _COMPILED_CREATE_BLOCK_MASK is None:
        _COMPILED_CREATE_BLOCK_MASK = torch.compile(create_block_mask)
    return _COMPILED_CREATE_BLOCK_MASK


def _get_flex_attention(device: torch.device):
    global _COMPILED_FLEX_ATTENTION
    if device.type != "cuda":
        return flex_attention
    if _COMPILED_FLEX_ATTENTION is None:
        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention)
    return _COMPILED_FLEX_ATTENTION


def build_tree_inputs(
    input_ids: torch.Tensor,
    document_ids: torch.Tensor,
    anchors: torch.Tensor,
    proposals: torch.Tensor,
    max_branches: int | None = None,
) -> dict:
    """Assemble base + branch tokens, position ids, and branch-valid masks."""
    base_ids = input_ids.reshape(-1)
    docs = document_ids.reshape(-1)
    if base_ids.numel() != docs.numel():
        raise ValueError("input_ids and document_ids must have equal length")
    if proposals.ndim != 2 or proposals.shape[0] != anchors.numel():
        raise ValueError("proposals must be [num_anchors, proposal_tokens]")
    num_valid = anchors.numel()
    if max_branches is None:
        max_branches = num_valid
    if max_branches < num_valid:
        raise ValueError(
            f"max_branches={max_branches} is smaller than num_anchors={num_valid}"
        )

    device = base_ids.device
    pad = max_branches - num_valid
    if pad:
        anchors_padded = torch.cat(
            [anchors, torch.zeros(pad, dtype=anchors.dtype, device=device)]
        )
        proposals_padded = torch.cat(
            [proposals, torch.zeros((pad, proposals.shape[1]), dtype=proposals.dtype, device=device)],
            dim=0,
        )
    else:
        anchors_padded = anchors
        proposals_padded = proposals
    valid_branches = torch.cat(
        [
            torch.ones(num_valid, dtype=torch.bool, device=device),
            torch.zeros(pad, dtype=torch.bool, device=device),
        ]
    )

    block_size = proposals_padded.shape[1] + 1
    branch_ids = torch.cat(
        [base_ids[anchors_padded].unsqueeze(1), proposals_padded], dim=1
    )
    tree_ids = torch.cat([base_ids, branch_ids.reshape(-1)]).unsqueeze(0)

    doc_starts = _context_starts(docs)
    base_len = base_ids.numel()
    base_positions = torch.arange(base_len, device=device) - doc_starts
    anchor_rel = anchors_padded - doc_starts[anchors_padded]
    branch_positions = (
        anchor_rel.unsqueeze(1)
        + torch.arange(block_size, device=device).unsqueeze(0)
    )
    tree_positions = torch.cat(
        [base_positions, branch_positions.reshape(-1)]
    ).unsqueeze(0)

    return {
        "tree_input_ids": tree_ids,
        "tree_position_ids": tree_positions,
        "base_ids": base_ids,
        "docs": docs,
        "base_len": base_len,
        "block_size": block_size,
        "num_valid": num_valid,
        "num_branches": max_branches,
        "anchors_padded": anchors_padded,
        "valid_branches": valid_branches,
        "branch_docs": docs[anchors_padded],
        "doc_starts": doc_starts,
    }


def create_tree_mask_mod(
    docs: torch.Tensor,
    base_len: int,
    anchors_padded: torch.Tensor,
    valid_branches: torch.Tensor,
    block_size: int,
):
    """Mask mod for base-causal + per-anchor causal branches.

    - Base queries attend causally to the same document.
    - Padding base tokens attend only to themselves (keeps their hidden finite
      so masked attention scores do not become NaN).
    - Branch queries attend to base tokens strictly before their anchor and to
      earlier tokens of the same branch.
    - Dummy padded branches attend only to their own branch.
    """

    branch_docs = docs[anchors_padded]

    def mask_mod(_b, _h, q_idx, kv_idx):
        q_is_base = q_idx < base_len
        kv_is_base = kv_idx < base_len
        q_base = q_idx.clamp(max=base_len - 1)
        kv_base = kv_idx.clamp(max=base_len - 1)
        q_doc_base = docs[q_base]
        kv_doc_base = docs[kv_base]

        base_real = (
            q_is_base
            & kv_is_base
            & (kv_idx <= q_idx)
            & (q_doc_base == kv_doc_base)
            & (q_doc_base != -1)
        )
        base_padding = q_is_base & kv_is_base & (q_idx == kv_idx) & (q_doc_base == -1)
        base_allow = base_real | base_padding

        q_branch = ((q_idx - base_len) // block_size).clamp(min=0)
        q_slot = (q_idx - base_len) % block_size
        q_valid = valid_branches[q_branch]
        q_anchor = anchors_padded[q_branch]
        q_branch_doc = branch_docs[q_branch]
        branch_base = (
            (~q_is_base)
            & kv_is_base
            & q_valid
            & (kv_idx < q_anchor)
            & (kv_doc_base == q_branch_doc)
            & (q_branch_doc != -1)
        )

        kv_branch = ((kv_idx - base_len) // block_size).clamp(min=0)
        kv_slot = (kv_idx - base_len) % block_size
        branch_same = (
            (~q_is_base)
            & (~kv_is_base)
            & (q_branch == kv_branch)
            & (kv_slot <= q_slot)
        )

        return base_allow | branch_base | branch_same

    return mask_mod


def build_tree_mask(tree: dict, device: torch.device, as_dense: bool = False):
    mask_mod = create_tree_mask_mod(
        tree["docs"],
        tree["base_len"],
        tree["anchors_padded"],
        tree["valid_branches"],
        tree["block_size"],
    )
    q_len = tree["base_len"] + tree["num_branches"] * tree["block_size"]
    kv_len = q_len
    if as_dense:
        q_idx = torch.arange(q_len, device=device).unsqueeze(1)
        kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)
        return mask_mod(0, 0, q_idx, kv_idx)
    create_mask = _get_create_block_mask(device)
    return create_mask(
        mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device=device
    )


def _dense_sdpa(q, k, v, mask, scale):
    mask = mask.unsqueeze(0).unsqueeze(0)
    try:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, enable_gqa=True
        )
    except TypeError:
        n_rep = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def verifier_tree_forward(
    verifier,
    tree: dict,
    mask,
    use_flex: bool = True,
):
    """Run the frozen Qwen3 verifier over the base+branch tree sequence."""
    model = verifier.model
    input_ids = tree["tree_input_ids"]
    position_ids = tree["tree_position_ids"]
    hidden = model.embed_tokens(input_ids)
    cos, sin = model.rotary_emb(hidden, position_ids)
    seq_len = hidden.shape[1]
    flex = _get_flex_attention(hidden.device) if use_flex else None

    for layer in model.layers:
        residual = hidden
        normed = layer.input_layernorm(hidden)
        attn = layer.self_attn
        q = attn.q_norm(
            attn.q_proj(normed).view(1, seq_len, -1, attn.head_dim)
        ).transpose(1, 2)
        k = attn.k_norm(
            attn.k_proj(normed).view(1, seq_len, -1, attn.head_dim)
        ).transpose(1, 2)
        v = attn.v_proj(normed).view(1, seq_len, -1, attn.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if use_flex:
            out = flex(
                q,
                k,
                v,
                block_mask=mask,
                scale=attn.scaling,
                enable_gqa=True,
            )
        else:
            out = _dense_sdpa(q, k, v, mask, attn.scaling)
        out = out.transpose(1, 2).reshape(1, seq_len, -1).contiguous()
        hidden = residual + attn.o_proj(out)
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))

    return model.norm(hidden)


def online_verifier_labels_tree(
    verifier,
    input_ids: torch.Tensor,
    document_ids: torch.Tensor,
    anchors: torch.Tensor,
    proposals: torch.Tensor,
    max_branches: int | None = None,
    use_flex: bool = True,
    return_hidden: bool = False,
):
    """Return greedy verifier labels in the same order as ``anchors``."""
    tree = build_tree_inputs(
        input_ids,
        document_ids,
        anchors,
        proposals,
        max_branches=max_branches,
    )
    mask = build_tree_mask(tree, tree["tree_input_ids"].device, as_dense=not use_flex)
    with torch.no_grad():
        hidden = verifier_tree_forward(verifier, tree, mask, use_flex=use_flex)
    base_len = tree["base_len"]
    block_size = tree["block_size"]
    num_branches = tree["num_branches"]
    num_valid = tree["num_valid"]
    branch_hidden = hidden[0, base_len:].reshape(
        num_branches, block_size, hidden.shape[-1]
    )[:num_valid, :LABEL_SLOTS]
    labels = verifier.lm_head(branch_hidden).argmax(dim=-1)
    if return_hidden:
        return labels, branch_hidden
    return labels
