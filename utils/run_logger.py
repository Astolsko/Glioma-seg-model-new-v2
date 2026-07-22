import csv
import json
import os
import sys
from datetime import datetime


class _Tee:
    """Writes to both the original stream and a log file, so every print()
    in the pipeline ends up in logs/<run>/log.txt as well as the console."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _sanitize_run_name(name):
    name = name.strip()
    for ch in ('/', '\\'):
        name = name.replace(ch, '_')
    return name


def _unique_run_dir(base_dir, name):
    candidate = os.path.join(base_dir, name)
    if not os.path.exists(candidate):
        return candidate
    i = 1
    while os.path.exists(f"{candidate}_{i}"):
        i += 1
    return f"{candidate}_{i}"


class RunLogger:
    """Owns everything about one training run's on-disk footprint:
    logs/<run_name>/
        log.txt               — full stdout/print capture
        config_snapshot.json  — cfg values used for this run
        metrics.csv           — one row per training epoch
        plots/                — data distribution + loss/dice/hd95/iou curves
        visualizations/       — pre-training qualitative sample checks
        attention/            — attention-map overlays saved during validation
        checkpoints/          — best_metric_model.pth
        testing/              — final test-set metrics + qualitative outputs
    """

    def __init__(self, run_name=None, base_dir="logs"):
        if not run_name:
            run_name = input("Enter a name for this run (used for the logs folder): ")
        run_name = _sanitize_run_name(run_name)
        if not run_name:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        self.run_dir = _unique_run_dir(base_dir, run_name)
        self.plots_dir = os.path.join(self.run_dir, "plots")
        self.vis_dir = os.path.join(self.run_dir, "visualizations")
        self.attention_dir = os.path.join(self.run_dir, "attention")
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.testing_dir = os.path.join(self.run_dir, "testing")
        self.testing_vis_dir = os.path.join(self.testing_dir, "visualizations")

        for d in (self.run_dir, self.plots_dir, self.vis_dir, self.attention_dir,
                  self.checkpoint_dir, self.testing_dir, self.testing_vis_dir):
            os.makedirs(d, exist_ok=True)

        self.log_path = os.path.join(self.run_dir, "log.txt")
        self.metrics_csv_path = os.path.join(self.run_dir, "metrics.csv")
        self.test_metrics_csv_path = os.path.join(self.testing_dir, "test_metrics.csv")
        self.config_snapshot_path = os.path.join(self.run_dir, "config_snapshot.json")
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "best_metric_model.pth")

        self._log_file = None
        self._stdout = None
        self._stderr = None
        self._metrics_writer = None
        self._metrics_file = None
        # The real terminal stream, kept undecorated (no log-file teeing) so
        # live-updating output (tqdm progress bars) can write straight to the
        # terminal without spamming log.txt with carriage-return redraws.
        self.console = None

    def __enter__(self):
        self._log_file = open(self.log_path, "a")
        self._stdout, self._stderr = sys.stdout, sys.stderr
        self.console = self._stdout
        sys.stdout = _Tee(self._stdout, self._log_file)
        sys.stderr = _Tee(self._stderr, self._log_file)
        print(f"Run started {datetime.now().isoformat(timespec='seconds')}, logging to {self.run_dir}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Run finished {datetime.now().isoformat(timespec='seconds')}")
        if self._metrics_file is not None:
            self._metrics_file.close()
        sys.stdout, sys.stderr = self._stdout, self._stderr
        self._log_file.close()
        return False

    def save_config_snapshot(self, cfg):
        def _to_plain(obj):
            if isinstance(obj, dict):
                return {k: _to_plain(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_plain(v) for v in obj]
            return obj

        with open(self.config_snapshot_path, "w") as f:
            json.dump(_to_plain(cfg), f, indent=2)
        print(f"Config snapshot saved: {self.config_snapshot_path}")

    def open_metrics_csv(self, fieldnames):
        self._metrics_file = open(self.metrics_csv_path, "w", newline="")
        self._metrics_writer = csv.DictWriter(self._metrics_file, fieldnames=fieldnames)
        self._metrics_writer.writeheader()

    def log_epoch_metrics(self, row: dict):
        self._metrics_writer.writerow(row)
        self._metrics_file.flush()

    def write_test_metrics(self, row: dict):
        with open(self.test_metrics_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        print(f"Test metrics saved: {self.test_metrics_csv_path}")
