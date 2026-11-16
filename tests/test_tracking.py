import json
import pytest
import torch
from pluto.train import run_training


def test_mlflow_resume_uses_one_run_and_logs_progress(tmp_path):
    mlflow = pytest.importorskip('mlflow')
    uri = 'sqlite:///' + str(tmp_path / 'tracking.db')
    run_dir = tmp_path / 'run'
    run_training(run_dir, epochs=2, seed=42, stop_after=1, tracking_uri=uri)
    first = torch.load(run_dir / 'checkpoint.pt', map_location='cpu', weights_only=False)
    run_training(run_dir, epochs=2, seed=42, resume=True, tracking_uri=uri)
    last = torch.load(run_dir / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert first['run_id'] == last['run_id']
    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    run = client.get_run(last['run_id'])
    assert len(client.get_metric_history(last['run_id'], 'train/loss')) == 2
    assert run.data.tags['progress/epoch'] == '2/2'
    assert 'T' in run.data.tags['progress/estimated_finish']
    sessions = list((run_dir / 'sessions').glob('*.json'))
    assert len(sessions) == 2
    assert json.loads(sessions[0].read_text())['config']['seed'] == 42
