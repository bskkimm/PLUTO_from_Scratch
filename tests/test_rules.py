import torch
from pluto.rules import score_rollouts


def test_rule_scoring_penalizes_collision_and_excludes_invalid():
    rollouts = torch.zeros(1, 2, 3, 4)
    rollouts[0, 0, :, 0] = torch.tensor([0.0, 1.0, 2.0])
    rollouts[0, 1, :, 0] = torch.tensor([0.0, 1.0, 2.0])
    rollouts[0, 1, :, 1] = 4.0
    valid = torch.tensor([[True, False]])
    obstacles = torch.tensor([[[1.0, 0.0]]])
    scores, terms = score_rollouts(
        rollouts, valid, obstacles=obstacles, collision_radius=0.5
    )
    assert terms["collision"][0, 0] and not terms["collision"][0, 1]
    assert scores[0, 0] == 0 and scores[0, 1] == 0


def test_rule_scoring_respects_drivable_mask():
    rollouts = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
            ]
        ]
    )
    sdf = torch.ones(1, 7, 7)
    sdf[0, 3, 4] = -1.0
    scores, terms = score_rollouts(
        rollouts, torch.ones(1, 2, dtype=torch.bool), sdf=sdf, resolution=1.0
    )
    assert not terms["drivable"].any()
    assert torch.equal(scores, torch.zeros_like(scores))


def test_model_candidate_selection_with_rollout_rules():
    from pluto.planner import select_trajectory
    from pluto.simulation import make_rollout_evaluator

    plans = torch.zeros(1, 1, 2, 3, 6)
    plans[..., 2] = 1.0
    plans[..., 4] = 1.0
    plans[0, 0, :, :, 0] = torch.tensor([0.1, 0.2, 0.3])
    plans[0, 0, 1, :, 1] = 5.0
    evaluator = make_rollout_evaluator(
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        lambda rollout, valid: score_rollouts(rollout, valid)[0],
    )
    selected, details = select_trajectory(
        plans,
        torch.tensor([[[2.0, 1.0]]]),
        torch.ones(1, 1, dtype=torch.bool),
        torch.zeros(1, 3, 6),
        evaluator,
        top_k=2,
    )
    assert selected.shape == (1, 3, 6)
    assert torch.isfinite(details["rule_score"]).all()
