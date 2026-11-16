"""Small deterministic synthetic training example; not the nuPlan recipe."""
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import time
import platform
import uuid
import torch
from .model import Pluto
from .recipes import get_recipe, make_scheduler
from .dataset import BatchManifest
from .losses import imitation_loss
from .augmentation import mask_agents, perturb_state
from .objective import contrastive_objective
from .synthetic import sample_batch
from .tracking import Tracker


def run_training(out_dir, epochs=3, seed=123, resume=False, stop_after=None, cil=False, manifest=None, tracking_uri=None, recipe='smoke'):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = get_recipe(recipe)
    cfg = settings.model
    data = BatchManifest(manifest) if manifest is not None else None
    if data is not None and cil:
        raise ValueError('manifest CIL requires validated triplet views, not synthetic augmentation')
    dataset_id = {'manifest_sha256': data.identity, 'split': data.split} if data else 'synthetic-v1'
    config = {'model': asdict(cfg), 'epochs': epochs, 'seed': seed, 'dataset': dataset_id, 'batch_size': 2 if data is None else 'from_manifest', 'lr': settings.learning_rate, 'weight_decay': settings.weight_decay, 'warmup_epochs': settings.warmup_epochs, 'recipe': recipe, 'cil': cil, 'tracking_uri': tracking_uri}
    ckpt_path = out_dir / 'checkpoint.pt'
    if ckpt_path.exists() and not resume:
        raise FileExistsError('checkpoint exists; use --resume or a new output directory')
    torch.manual_seed(seed)
    model = Pluto(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = make_scheduler(optimizer, epochs, settings.warmup_epochs)
    start = 0
    run_id = None
    if resume:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if ckpt['config'] != config:
            raise ValueError('resume config differs from checkpoint config')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        torch.set_rng_state(ckpt['rng'])
        start = ckpt['epoch']
        run_id = ckpt.get('run_id')
    tracker = Tracker(tracking_uri, config, run_id) if tracking_uri else None
    run_id = tracker.run_id if tracker else None
    session_dir = out_dir / 'sessions'
    session_dir.mkdir(exist_ok=True)
    session = {'config': config, 'resume_epoch': start, 'run_id': run_id, 'started_at': datetime.now().astimezone().isoformat(),
               'python': platform.python_version(), 'torch': torch.__version__}
    (session_dir / (uuid.uuid4().hex + '.json')).write_text(json.dumps(session, indent=2) + '\n')
    synthetic_batch = sample_batch(batch_size=2, future_steps=cfg.future_steps, seed=seed) if data is None else None
    start_time = time.monotonic()
    limit = min(epochs, stop_after if stop_after is not None else epochs)
    for epoch in range(start, limit):
        model.train()
        epoch_losses = []
        batches = (data[i] for i in range(len(data))) if data is not None else (synthetic_batch,)
        for batch in batches:
            if batch['target'].shape[1] != cfg.future_steps:
                raise ValueError('batch target horizon differs from the chosen recipe')
            optimizer.zero_grad()
            if cil:
                positive = perturb_state(batch, torch.randn_like(batch['ego_state']) * 0.05)
                drop = torch.zeros_like(batch['agent_mask'][..., 0])
                drop[:, -1] = True
                negative = mask_agents(batch, drop)
                # Toy views are diagnostics, not validated causal nuPlan augmentations.
                loss = contrastive_objective(model, batch, positive, negative)['total']
            else:
                loss = imitation_loss(model(batch), batch)['total']
            if not torch.isfinite(loss):
                raise RuntimeError('non-finite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            epoch_losses.append(loss.item())
        scheduler.step()
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'rng': torch.get_rng_state(),
                 'epoch': epoch + 1, 'config': config, 'run_id': run_id}
        temporary = out_dir / 'checkpoint.pt.tmp'
        torch.save(state, temporary)
        os.replace(temporary, ckpt_path)
        elapsed = time.monotonic() - start_time
        remaining = elapsed / (epoch + 1 - start) * (epochs - epoch - 1)
        finish = datetime.now().astimezone() + timedelta(seconds=remaining)
        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        if tracker:
            tracker.progress(epoch + 1, epochs, epoch_loss, optimizer.param_groups[0]['lr'], elapsed, remaining, finish.isoformat(), epoch == start)
        print(json.dumps({'epoch': epoch + 1, 'epochs': epochs, 'loss': epoch_loss,
                          'active_seconds': elapsed, 'remaining_seconds': remaining,
                          'estimated_finish': finish.isoformat(), 'provisional': epoch == start}))
    if tracker and limit == epochs and start < epochs:
        tracker.finish()
    return ckpt_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runs/smoke')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--cil', action='store_true', help='enable diagnostic synthetic contrastive training')
    parser.add_argument('--manifest', help='local preprocessed tensor-batch manifest')
    parser.add_argument('--mlflow-uri', help='opt-in MLflow tracking URI (e.g. sqlite:///runs/mlflow.db)')
    parser.add_argument('--recipe', choices=['smoke', 'practical', 'paper_shape'], default='smoke')
    args = parser.parse_args()
    if args.epochs < 1 or (args.stop_after is not None and args.stop_after < 1):
        parser.error('epochs and stop-after must be positive')
    run_training(args.out, args.epochs, args.seed, args.resume, args.stop_after, args.cil, args.manifest, args.mlflow_uri, args.recipe)


if __name__ == '__main__':
    main()
