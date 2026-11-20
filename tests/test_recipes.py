import pytest
import torch
from pluto.model import Pluto
from pluto.recipes import get_recipe, make_optimizer, make_scheduler


def test_recipe_dimensions_and_unknown_name():
    smoke = get_recipe("smoke")
    paper = get_recipe("paper_shape")
    assert (smoke.model.dim, smoke.model.future_steps) == (16, 4)
    assert (
        paper.model.dim,
        paper.model.encoder_layers,
        paper.model.decoder_layers,
        paper.model.modes,
        paper.model.future_steps,
    ) == (128, 4, 4, 12, 80)
    assert paper.warmup_epochs == 3
    assert paper.min_learning_rate == 1e-6
    with pytest.raises(ValueError):
        get_recipe("does-not-exist")


def test_warmup_then_cosine_scheduler():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = make_scheduler(optimizer, epochs=6, warmup_epochs=3)
    rates = []
    for _ in range(6):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] < rates[1] < rates[2] <= rates[3]
    assert rates[3] > rates[4] > rates[5]


def test_optimizer_decay_groups_cover_trainable_parameters_once():
    recipe = get_recipe("smoke")
    model = Pluto(recipe.model)
    model.free_head[0].weight.requires_grad_(False)
    optimizer = make_optimizer(model, recipe.learning_rate, recipe.weight_decay)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    decay, no_decay = (set(group["param_names"]) for group in optimizer.param_groups)
    assert "ego_encoder.attention.in_proj_weight" in decay
    assert "ego_encoder.attention.out_proj.weight" in decay
    assert "agent_encoder.tokenizer.weight" in decay
    assert "ego_encoder.query" in no_decay
    assert "ego_encoder.position" in no_decay
    assert "scene_position.frequency" in no_decay
    assert "agent_type.weight" in no_decay
    assert "ego_encoder.attention.in_proj_bias" in no_decay
    assert "decoder.0.norms.0.weight" in no_decay
    for group in optimizer.param_groups:
        assert group["param_names"] == sorted(group["param_names"])
        assert group["weight_decay"] == (
            recipe.weight_decay if group is optimizer.param_groups[0] else 0.0
        )


def test_optimizer_decay_exclusions_change_the_actual_update():
    model = torch.nn.ModuleDict(
        {
            "linear": torch.nn.Linear(2, 2),
            "norm": torch.nn.LayerNorm(2),
            "embedding": torch.nn.Embedding(2, 2),
        }
    )
    model.register_parameter("query", torch.nn.Parameter(torch.ones(2)))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(1.0)
            parameter.grad = torch.zeros_like(parameter)
    optimizer = make_optimizer(model, learning_rate=0.1, weight_decay=0.2)
    optimizer.step()
    for name, parameter in model.named_parameters():
        expected = 0.98 if name == "linear.weight" else 1.0
        torch.testing.assert_close(parameter, torch.full_like(parameter, expected))


def test_scheduler_matches_released_warmup_cosine_and_floor():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = make_scheduler(optimizer, epochs=6, warmup_epochs=3, min_lr=1e-6)
    rates = []
    for _ in range(8):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates == pytest.approx(
        [1e-3 / 3, 2e-3 / 3, 1e-3, 1e-3, 0.00075025, 0.00025075, 1e-6, 1e-6]
    )


def test_scheduler_without_warmup_and_with_different_group_rates():
    optimizer = torch.optim.AdamW(
        [
            {"params": [torch.nn.Parameter(torch.tensor(0.0))], "lr": 1e-3},
            {"params": [torch.nn.Parameter(torch.tensor(0.0))], "lr": 2e-3},
        ]
    )
    scheduler = make_scheduler(optimizer, epochs=2, warmup_epochs=0, min_lr=1e-6)
    optimizer.step()
    scheduler.step()
    assert [g["lr"] for g in optimizer.param_groups] == pytest.approx(
        [0.0005005, 0.0010005]
    )
    optimizer.step()
    scheduler.step()
    assert [g["lr"] for g in optimizer.param_groups] == pytest.approx([1e-6, 1e-6])


@pytest.mark.parametrize("min_lr", [-1.0, float("nan"), 0.01])
def test_scheduler_rejects_invalid_minimum(min_lr):
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.tensor(0.0))], lr=1e-3)
    with pytest.raises(ValueError, match="minimum learning rate"):
        make_scheduler(optimizer, epochs=2, warmup_epochs=0, min_lr=min_lr)
