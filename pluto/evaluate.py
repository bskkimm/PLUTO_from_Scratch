"""Report a synthetic open-loop diagnostic, not the nuPlan benchmark."""
import argparse
import torch
from .model import Pluto, PlutoConfig
from .synthetic import sample_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = PlutoConfig(**state['config']['model'])
    model = Pluto(cfg)
    model.load_state_dict(state['model'])
    model.eval()
    batch = sample_batch(2, cfg.future_steps, seed=state['config']['seed'])
    with torch.no_grad():
        out = model(batch)
        ade = (out['selected'][..., :2] - batch['target'][..., :2]).norm(dim=-1).mean()
    print(f'synthetic selected ADE (m): {ade.item():.6f}; not a nuPlan metric')


if __name__ == '__main__':
    main()
