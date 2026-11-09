"""Single-block objectives. Log probabilities describe seven independent positions."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def accepted_length(tokens, target_prefix):
    """Exact greedy-verifier reward; tokens may have a candidate dimension."""
    if tokens.shape[-1] != 7 or target_prefix.shape[-1] != 7:
        raise ValueError("Only block_size=8 (seven draft positions) is supported")
    return 1 + tokens.eq(target_prefix).long().cumprod(-1).sum(-1)


def policy_gradient(sequence_logp, rewards):
    """Batch-mean baseline and detached batch standard deviation scaling.

    Includes the sampled action in the baseline, as explicitly requested.
    This normalization is a practical surrogate, not an unbiased estimator.
    Use the full sequence score; masking by observed failure can bias it.
    """
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError("Need at least two rewards for a batch baseline")
    if sequence_logp.shape != rewards.shape:
        raise ValueError("One sequence score per reward is required")
    rewards = rewards.detach().float()
    centered = rewards - rewards.mean()
    advantage = centered / rewards.std(unbiased=False).clamp_min(1e-6)
    return -(advantage * sequence_logp).mean(), advantage


def pairwise_ranking(sequence_logp, rewards):
    """Gap-weighted pairwise logistic loss, ignoring ties.

    Candidate pairs belong to the same context. This is a ranking surrogate,
    not the policy gradient of expected accepted length.
    """
    if sequence_logp.ndim != 2 or sequence_logp.shape[-1] != 2:
        raise ValueError("Expected [contexts, two candidates]")
    if sequence_logp.shape != rewards.shape:
        raise ValueError("Candidate scores and rewards must have equal shapes")
    gap = (rewards[:, 0] - rewards[:, 1]).detach().float()
    margin = sequence_logp[:, 0] - sequence_logp[:, 1]
    return (gap.abs() * F.softplus(-gap.sign() * margin)).mean()


def eal_with_reference_kl(logits, labels, reference_logits, beta=0.1):
    """Negative expected accepted draft count + beta KL(frozen reference || draft).

    Reference is the common starting draft checkpoint, not target logits.
    KL is averaged over positions; EAL is summed over seven prefix lengths.
    """
    if logits.shape != reference_logits.shape or logits.shape[-2] != 7:
        raise ValueError("Expected matching [..., 7, vocabulary] tensors")
    if beta < 0:
        raise ValueError("KL coefficient must be nonnegative")
    logp = logits.float().log_softmax(-1)
    label_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    eal_loss = -label_logp.cumsum(-1).exp().sum(-1).mean()
    ref = reference_logits.detach().float().log_softmax(-1)
    kl = (ref.exp() * (ref - logp)).sum(-1).mean()
    return eal_loss + beta * kl, eal_loss, kl


def soft_eal_loss(logits, labels, valid_mask=None):
    """Differentiable EAL on verifier labels from an online draft prefix.

    ``labels`` are expected to come from a verifier forward conditioned on the
    draft's actual prefix.  Unlike the original fixed seven-slot helper this
    accepts partial blocks through ``valid_mask`` and keeps F1's soft prefix
    weighting (cumulative products of draft probabilities).
    """
    if logits.ndim < 2 or logits.shape[-2] != 7:
        raise ValueError("Expected [..., 7, vocabulary] logits")
    if labels.shape != logits.shape[:-1]:
        raise ValueError("labels must match logits without vocabulary dimension")
    if valid_mask is None:
        valid_mask = torch.ones_like(labels, dtype=torch.bool)
    if valid_mask.shape != labels.shape:
        raise ValueError("valid_mask must match labels")
    logp = logits.float().log_softmax(-1)
    token_logp = logp.gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)
    prefix = token_logp.cumsum(-1).exp()
    # A partial final block contributes only through its valid prefix.  Masks
    # after the first invalid slot are excluded from the EAL sum.
    prefix_valid = valid_mask.long().cumprod(-1).bool()
    eal = (prefix * prefix_valid).sum(-1)
    return -eal.mean()


def _soft_eal_chunk(logits, labels, valid_mask):
    logp = logits.float().log_softmax(-1)
    token_logp = logp.gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)
    prefix = token_logp.cumsum(-1).exp()
    prefix_valid = valid_mask.long().cumprod(-1).bool()
    return (prefix * prefix_valid).sum(-1)


def soft_eal_loss_chunked(
    logits,
    labels,
    valid_mask=None,
    chunk_blocks=256,
    log_eal=False,
    eps=1e-6,
):
    """Same objective as :func:`soft_eal_loss` without a full fp32 vocabulary.

    The full-width fp32 log-softmax costs ``num_blocks * 7 * vocab * 4`` bytes
    and its autograd graph retains it until backward.  At 3072 DFlash anchors
    that is ~15 GB before activations.  This variant checkpoints one block
    chunk at a time, so only the bf16 logits (already owned by the model) plus
    one chunk's fp32 softmax are resident.  Values and gradients match
    :func:`soft_eal_loss`.

    With ``log_eal=True`` the batch mean EAL is passed through ``-log``
    (``-log(mean_eal + eps)``).  This is a strictly monotone transform of the
    mean EAL; it changes the global gradient scale but not the per-position
    prefix weights.
    """
    if logits.ndim < 2 or logits.shape[-2] != 7:
        raise ValueError("Expected [..., 7, vocabulary] logits")
    if labels.shape != logits.shape[:-1]:
        raise ValueError("labels must match logits without vocabulary dimension")
    if valid_mask is None:
        valid_mask = torch.ones_like(labels, dtype=torch.bool)
    if valid_mask.shape != labels.shape:
        raise ValueError("valid_mask must match labels")
    if chunk_blocks <= 0:
        raise ValueError("chunk_blocks must be positive")
    if log_eal and eps <= 0:
        raise ValueError("log-EAL eps must be positive")

    num_blocks = logits.shape[0]
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, num_blocks, chunk_blocks):
        end = min(start + chunk_blocks, num_blocks)
        chunk_sum = checkpoint(
            _soft_eal_chunk,
            logits[start:end],
            labels[start:end],
            valid_mask[start:end],
            use_reentrant=False,
        )
        total = total + chunk_sum.sum()
    mean_eal = total / max(num_blocks, 1)
    if log_eal:
        return -torch.log(mean_eal + eps)
    return -mean_eal


def prefix_weights_logspace(
    logits, labels, valid_mask=None, chunk_blocks=256
):
    """Normalized per-slot weights proportional to ``sum_{i>=k} P_i``.

    ``P_i = prod_{j<=i} p_j(label_j)`` is the draft probability that the
    verifier prefix reaches position i.  Weights are computed in log space so
    that the very small early-training products do not underflow to zero, and
    are detached because the EAL term already differentiates through them.
    """
    if logits.ndim < 2 or logits.shape[-2] != 7:
        raise ValueError("Expected [..., 7, vocabulary] logits")
    if labels.shape != logits.shape[:-1]:
        raise ValueError("labels must match logits without vocabulary dimension")
    if valid_mask is None:
        valid_mask = torch.ones_like(labels, dtype=torch.bool)
    if valid_mask.shape != labels.shape:
        raise ValueError("valid_mask must match labels")
    if chunk_blocks <= 0:
        raise ValueError("chunk_blocks must be positive")

    num_blocks = logits.shape[0]
    token_logp = torch.empty(
        labels.shape, dtype=torch.float32, device=logits.device
    )
    for start in range(0, num_blocks, chunk_blocks):
        end = min(start + chunk_blocks, num_blocks)
        logp = logits[start:end].float().log_softmax(-1)
        token_logp[start:end] = logp.gather(
            -1, labels[start:end].unsqueeze(-1)
        ).squeeze(-1)
    prefix_valid = valid_mask.long().cumprod(-1).bool()
    token_logp = token_logp.masked_fill(~prefix_valid, float("-inf"))
    log_prefix = token_logp.cumsum(-1)
    log_weights = torch.logcumsumexp(log_prefix.flip(-1), dim=-1).flip(-1)
    log_weights = log_weights.masked_fill(~prefix_valid, float("-inf"))
    log_norm = torch.logsumexp(log_weights, dim=-1, keepdim=True)
    weights = torch.exp(log_weights - log_norm)
    weights = torch.where(
        torch.isfinite(log_norm), weights, torch.zeros_like(weights)
    )
    return weights.detach()


def _forward_kl_chunk(
    logits, target_hidden, target_lm_head, valid_mask, weights, temperature
):
    with torch.no_grad():
        weight = getattr(target_lm_head, "weight", None)
        if weight is not None:
            target_hidden = target_hidden.to(dtype=weight.dtype)
        target_logits = target_lm_head(target_hidden)
        q_log = (target_logits.float() / temperature).log_softmax(-1)
    logp = (logits.float() / temperature).log_softmax(-1)
    kl = (q_log.exp() * (q_log - logp)).sum(-1)
    kl = kl * valid_mask.float()
    return (kl * weights).sum(), (weights * valid_mask.float()).sum()


def forward_kl_chunked(
    logits,
    target_hidden,
    target_lm_head,
    valid_mask=None,
    weights=None,
    chunk_blocks=256,
    temperature=1.0,
):
    """Prefix-weighted forward KL ``KL(verifier || draft)`` per slot.

    ``target_lm_head`` is applied to ``target_hidden`` in the same block/slot
    layout as ``logits``.  The target side is no-grad and recomputed inside the
    checkpointed chunk, so the full ``[blocks, 7, vocab]`` target logits are
    never resident.
    """
    if logits.shape[-2] != 7:
        raise ValueError("Expected [..., 7, vocabulary] logits")
    if target_hidden.shape[:-1] != logits.shape[:-1]:
        raise ValueError("target_hidden must match logits without vocabulary dim")
    if valid_mask is None:
        valid_mask = torch.ones_like(logits[..., 0], dtype=torch.bool)
    if valid_mask.shape != logits.shape[:-1]:
        raise ValueError("valid_mask must match logits without vocabulary dim")
    if weights is None:
        weights = valid_mask.float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1.0)
    if weights.shape != valid_mask.shape:
        raise ValueError("weights must match valid_mask")
    if chunk_blocks <= 0:
        raise ValueError("chunk_blocks must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    num_blocks = logits.shape[0]
    total = torch.zeros((), device=logits.device, dtype=torch.float32)
    norm = torch.zeros((), device=logits.device, dtype=torch.float32)
    for start in range(0, num_blocks, chunk_blocks):
        end = min(start + chunk_blocks, num_blocks)
        chunk_total, chunk_norm = checkpoint(
            _forward_kl_chunk,
            logits[start:end],
            target_hidden[start:end],
            target_lm_head,
            valid_mask[start:end],
            weights[start:end],
            temperature,
            use_reentrant=False,
        )
        total = total + chunk_total
        norm = norm + chunk_norm
    return total / norm.clamp_min(1e-9)


def first_error_margin(logits, labels, margin=0.5, prefix_weight=0.1):
    """Deterministic first-error repair; no auxiliary gradient after rejection.

    Weak margin protection applies to the accepted prefix (all seven slots
    if no error). KL regularization is deliberately handled separately.
    """
    if logits.shape[-2] != 7 or labels.shape != logits.shape[:-1]:
        raise ValueError('Expected [..., 7, vocabulary] and [..., 7] labels')
    if margin < 0 or not 0 <= prefix_weight <= 1:
        raise ValueError('Invalid margin or prefix weight')
    correct = logits.detach().argmax(-1).eq(labels)
    reached = torch.cat((torch.ones_like(correct[..., :1]),
                         correct[..., :-1].long().cumprod(-1).bool()), -1)
    weights = reached.float() * torch.where(correct, prefix_weight, 1.0)
    positive = logits.float().gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    negative = logits.float().scatter(-1, labels.unsqueeze(-1), -torch.inf).amax(-1)
    penalty = F.relu(margin + negative - positive)
    return ((penalty * weights).sum(-1) / weights.sum(-1).clamp_min(1)).mean()
