"""Simulator-independent candidate selection boundary (paper Algorithm 2)."""

import torch


def select_trajectory(
    trajectory, logits, reference_valid, free, evaluator, top_k=20, alpha=0.3
):
    """Select original top-K proposal using an externally supplied rule evaluator.

    `trajectory` [B,R,M,T,C], `logits` [B,R,M], reference_valid [B,R], and
    free [B,T,C] are in the same frame. evaluator(proposals, valid) returns
    [B,K] rule scores after any desired simulation. Invalid entries must be
    ignored by the evaluator. This function does not implement a simulator,
    collision checking, or official nuPlan scoring. It returns the original
    proposal, not the simulated rollout; no-valid-reference rows use `free`.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    b, r, m, t, c = trajectory.shape
    if r == 0:
        empty = logits.new_empty((b, 0))
        return free, {
            "candidate_indices": empty.long(),
            "candidate_valid": empty.bool(),
            "learned_probability": empty,
            "rule_score": empty,
        }
    valid = reference_valid.unsqueeze(-1).expand(b, r, m).reshape(b, r * m)
    scores = logits.reshape(b, r * m).masked_fill(~valid, float("-inf"))
    k = min(top_k, r * m)
    values, indices = scores.topk(k, dim=-1)
    candidate_valid = valid.gather(1, indices)
    # Softmax needs at least one finite entry even for an all-empty example.
    safe_values = values.masked_fill(~candidate_valid, -1e9)
    probability = safe_values.softmax(-1) * candidate_valid.to(values.dtype)
    probability = probability / probability.sum(-1, keepdim=True).clamp(min=1e-12)
    flat = trajectory.reshape(b, r * m, t, c)
    proposals = flat.gather(1, indices[:, :, None, None].expand(b, k, t, c))
    rule_score = evaluator(proposals, candidate_valid)
    if (
        rule_score.shape != values.shape
        or not torch.isfinite(rule_score[candidate_valid]).all()
    ):
        raise ValueError(
            "evaluator must return finite [B,K] scores for valid candidates"
        )
    final = (rule_score + alpha * probability).masked_fill(
        ~candidate_valid, float("-inf")
    )
    best = final.argmax(-1)
    selected = proposals[torch.arange(b, device=trajectory.device), best]
    selected = torch.where(candidate_valid.any(-1)[:, None, None], selected, free)
    return selected, {
        "candidate_indices": indices,
        "candidate_valid": candidate_valid,
        "learned_probability": probability,
        "rule_score": rule_score,
    }
