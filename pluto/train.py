"""Small deterministic synthetic training example; not the nuPlan recipe."""
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import time
import torch
from .model import Pluto, PlutoConfig
from .losses import imitation_loss
from .synthetic import sample_batch


def run_training(out_dir, epochs=3, seed=123, resume=False, stop_after=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = PlutoConfig(dim=16, heads=4, encoder_layers=1, decoder_layers=1, modes=3, future_steps=4, dropout=0.1)
    config = {'model': asdict(cfg), 'epochs': epochs, 'seed': seed, 'dataset': 'synthetic-v1', 'batch_size': 2, 'lr': 0.001}
    ckpt_path = out_dir / 'checkpoint.pt'
    if ckpt_path.exists() and not resume:
        raise FileExistsError('checkpoint exists; use --resume or a new output directory')
    torch.manual_seed(seed)
    model = Pluto(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    start = 0
    if resume:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if ckpt['config'] != config:
            raise ValueError('resume config differs from checkpoint config')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        torch.set_rng_state(ckpt['rng'])
        start = ckpt['epoch']
    batch = sample_batch(batch_size=2, future_steps=cfg.future_steps, seed=seed)
    start_time = time.monotonic()
    limit = min(epochs, stop_after if stop_after is not None else epochs)
    for epoch in range(start, limit):
        model.train()
        optimizer.zero_grad()
        loss = imitation_loss(model(batch), batch)['total']
        if not torch.isfinite(loss):
            raise RuntimeError('non-finite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        scheduler.step()
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'rng': torch.get_rng_state(),
                 'epoch': epoch + 1, 'config': config}
        temporary = out_dir / 'checkpoint.pt.tmp'
        torch.save(state, temporary)
        os.replace(temporary, ckpt_path)
        elapsed = time.monotonic() - start_time
        remaining = elapsed / (epoch + 1 - start) * (epochs - epoch - 1)
        finish = datetime.now().astimezone() + timedelta(seconds=remaining)
        print(json.dumps({'epoch': epoch + 1, 'epochs': epochs, 'loss': loss.item(),
                          'active_seconds': elapsed, 'remaining_seconds': remaining,
                          'estimated_finish': finish.isoformat(), 'provisional': epoch == start}))
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs/smoke')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.epochs < 1 or (args.stop_after is not None and args.stop_after < 1):
        parser.error('epochs and stop-after must be positive')
    run_training(args.out, args.epochs, args.seed, args.resume, args.stop_after)


if __name__ == '__main__':
    main()
