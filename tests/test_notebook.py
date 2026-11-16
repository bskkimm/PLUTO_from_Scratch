import json
from pathlib import Path
import torch
from pluto.model import Pluto, PlutoConfig
from pluto.losses import imitation_loss
from pluto.synthetic import sample_batch


def test_notebook_runs_and_matches_package():
    notebook = json.loads(Path('notebooks/pluto_e2e_walkthrough.ipynb').read_text())
    namespace = {}
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None and cell['outputs'] == []
            exec(compile(''.join(cell['source']), cell['id'], 'exec'), namespace)
    cfg = PlutoConfig(dim=16, heads=4, encoder_layers=1, decoder_layers=1, modes=3, future_steps=4, dropout=0.)
    production = Pluto(cfg)
    notebook_model = namespace['Pluto'](namespace['PlutoConfig'](**vars(cfg)))
    notebook_model.load_state_dict(production.state_dict())
    production.eval(); notebook_model.eval()
    batch = sample_batch(2, 4)
    a, b = production(batch), notebook_model(batch)
    assert torch.allclose(a['trajectory'], b['trajectory'])
    la = imitation_loss(a, batch)['total']
    lb = namespace['imitation_loss'](b, batch)['total']
    la.backward(); lb.backward()
    assert torch.allclose(la, lb)
    for p, q in zip(production.parameters(), notebook_model.parameters()):
        if p.grad is not None:
            assert torch.allclose(p.grad, q.grad)
