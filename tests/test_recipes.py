import pytest
import torch
from pluto.recipes import get_recipe, make_scheduler


def test_recipe_dimensions_and_unknown_name():
    smoke = get_recipe('smoke')
    paper = get_recipe('paper_shape')
    assert (smoke.model.dim, smoke.model.future_steps) == (16, 4)
    assert (paper.model.dim, paper.model.encoder_layers, paper.model.decoder_layers, paper.model.modes, paper.model.future_steps) == (128, 4, 4, 12, 80)
    assert paper.warmup_epochs == 3
    with pytest.raises(ValueError):
        get_recipe('does-not-exist')


def test_warmup_then_cosine_scheduler():
    parameter = torch.nn.Parameter(torch.tensor(0.))
    optimizer = torch.optim.AdamW([parameter], lr=1.)
    scheduler = make_scheduler(optimizer, epochs=6, warmup_epochs=3)
    rates = []
    for _ in range(6):
        rates.append(optimizer.param_groups[0]['lr'])
        optimizer.step(); scheduler.step()
    assert rates[0] < rates[1] < rates[2] <= rates[3]
    assert rates[3] > rates[4] > rates[5]
