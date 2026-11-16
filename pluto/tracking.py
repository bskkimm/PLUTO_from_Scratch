"""Optional local or explicitly configured MLflow tracking."""

import json


class Tracker:
    def __init__(self, uri, config, run_id=None):
        try:
            from mlflow.tracking import MlflowClient
        except ImportError as exc:
            raise RuntimeError(
                "MLflow tracking requested; install mlflow to use it"
            ) from exc
        self.client = MlflowClient(tracking_uri=uri)
        experiment = self.client.get_experiment_by_name("pluto-from-scratch")
        experiment_id = (
            experiment.experiment_id
            if experiment
            else self.client.create_experiment("pluto-from-scratch")
        )
        if run_id is None:
            run_id = self.client.create_run(
                experiment_id, tags={"dataset": str(config["dataset"])}
            ).info.run_id
            self.client.log_param(
                run_id, "resolved_config", json.dumps(config, sort_keys=True)
            )
        else:
            self.client.get_run(run_id)
        self.run_id = run_id

    def progress(self, epoch, total, loss, lr, timing, components):
        for name, value in {
            "train/loss": loss,
            "train/lr": lr,
            "timing/session_active_seconds": timing["session_active_seconds"],
            "timing/total_active_seconds": timing["total_active_seconds"],
            "timing/wall_seconds": timing["wall_seconds"],
            "timing/remaining_seconds": timing["remaining_seconds"],
        }.items():
            self.client.log_metric(self.run_id, name, float(value), step=epoch)
        for name, value in components.items():
            self.client.log_metric(
                self.run_id, f"train/{name}", float(value), step=epoch
            )
        for name, value in {
            "progress/epoch": f"{epoch}/{total}",
            "progress/estimated_finish": timing["estimated_finish"],
            "progress/provisional": str(timing["provisional"]).lower(),
            "progress/updated_at": timing["updated_at"],
        }.items():
            self.client.set_tag(self.run_id, name, value)

    def finish(self):
        self.client.set_terminated(self.run_id, status="FINISHED")
