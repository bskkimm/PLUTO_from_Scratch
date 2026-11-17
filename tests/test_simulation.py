import torch
from pluto.simulation import bicycle_step, lqr_rollout


def test_bicycle_step_straight_and_stationary():
    state = torch.tensor([[0.0, 0.0, 0.0, 2.0], [1.0, 2.0, 1.0, 0.0]])
    action = torch.zeros(2, 2)
    result = bicycle_step(state, action, dt=0.5, wheelbase=2.5)
    assert torch.allclose(result[0], torch.tensor([1.0, 0.0, 0.0, 2.0]))
    assert torch.allclose(result[1], state[1])


def test_lqr_rollout_tracks_feasible_straight_reference():
    initial = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    trajectory = torch.zeros(1, 1, 5, 6)
    trajectory[0, 0, :, 0] = torch.arange(1.0, 6.0) * 0.1
    trajectory[..., 2] = 1.0
    trajectory[..., 4] = 1.0
    rollout = lqr_rollout(initial, trajectory, dt=0.1, wheelbase=2.5)
    assert rollout.shape == (1, 1, 6, 4)
    assert torch.allclose(rollout[0, 0, 1:, 0], trajectory[0, 0, :, 0], atol=1e-5)
    assert torch.isfinite(rollout).all()


def test_rollout_evaluator_passes_simulated_states_to_rule_scorer():
    from pluto.simulation import make_rollout_evaluator

    initial = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    trajectory = torch.zeros(1, 2, 3, 6)
    trajectory[..., 2] = 1.0
    trajectory[..., 4] = 1.0
    trajectory[..., 0] = torch.tensor([0.1, 0.2, 0.3])
    valid = torch.tensor([[True, False]])

    def scorer(rollouts, mask):
        assert rollouts.shape == (1, 2, 4, 4)
        assert torch.equal(mask, valid)
        return rollouts[..., -1, 0]

    evaluator = make_rollout_evaluator(initial, scorer)
    assert torch.allclose(
        evaluator(trajectory, valid)[0, 0], torch.tensor(0.3), atol=1e-5
    )
