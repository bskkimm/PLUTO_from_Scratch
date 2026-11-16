import torch
from pluto.planner import select_trajectory


def test_rule_score_selects_topk_without_modifying_trajectory():
    trajectory = torch.zeros(1, 1, 3, 2, 6)
    trajectory[0, 0, :, :, 0] = torch.tensor([0.0, 1.0, 2.0])[:, None]
    logits = torch.tensor([[[3.0, 2.0, 1.0]]])

    def evaluator(proposals, valid):
        return proposals[..., 0, 0] * 2

    selected, info = select_trajectory(
        trajectory,
        logits,
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 2, 6),
        evaluator,
        top_k=2,
        alpha=0.3,
    )
    assert torch.equal(selected, trajectory[:, 0, 1])
    assert info["candidate_indices"].tolist() == [[0, 1]]
    assert torch.allclose(info["learned_probability"].sum(-1), torch.ones(1))


def test_reference_mask_and_empty_reference_fallback():
    trajectory = torch.randn(2, 2, 2, 3, 6)
    logits = torch.tensor([[[0.0, 0.0], [100.0, 100.0]], [[3.0, 3.0], [4.0, 4.0]]])
    mask = torch.tensor([[True, False], [False, False]])
    free = torch.randn(2, 3, 6)
    selected, info = select_trajectory(
        trajectory,
        logits,
        mask,
        free,
        lambda x, valid: torch.zeros_like(valid, dtype=x.dtype),
        top_k=3,
    )
    assert torch.equal(selected[1], free[1])
    assert info["candidate_indices"][0, info["candidate_valid"][0]].max() < 2
    assert torch.isfinite(info["learned_probability"]).all()
