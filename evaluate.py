"""Re-evaluate a finished run's checkpoint with the current inference recipe —
no retraining.

    python evaluate.py --run v1-run3                    # test-set eval
    python evaluate.py --run v1-run3 --tune-thresholds  # tune on val first

Everything it changes lives at inference time: sliding-window overlap and
blending, flip-TTA, per-channel probability thresholds, and connected-component
cleanup (see cfg.infer). Results land in logs/<run>/eval/ so the original run's
artifacts are left untouched.

`--tune-thresholds` searches the per-channel threshold on the VALIDATION split
and then applies the winner to the test split. Tuning and reporting on the same
split would be tuning on the test set.
"""
import argparse
import os

from utils.env_check import ensure_dependencies
ensure_dependencies()

from config import cfg
from utils.dataloader import build_dataloaders
from utils.engine import build_model, get_device, run_inference, run_test
from utils.postprocess import save_infer_config, tune_and_save


class _EvalPaths:
    """The slice of RunLogger's interface that run_test actually touches.

    run_test only ever reads .checkpoint_path / .testing_vis_dir / .console and
    calls .write_test_metrics, so re-evaluating an existing run needs this, not
    a whole second RunLogger (which would mint a fresh logs/<name>_1 folder and
    scatter the results away from the run they describe).
    """

    def __init__(self, run_dir, tag):
        self.checkpoint_path = os.path.join(run_dir, "checkpoints", "best_metric_model.pth")
        self.eval_dir = os.path.join(run_dir, "eval")
        self.testing_vis_dir = os.path.join(self.eval_dir, "visualizations")
        self.test_metrics_csv_path = os.path.join(self.eval_dir, f"test_metrics_{tag}.csv")
        self.console = None
        for d in (self.eval_dir, self.testing_vis_dir):
            os.makedirs(d, exist_ok=True)

    def write_test_metrics(self, row):
        import csv
        with open(self.test_metrics_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        print(f"Test metrics saved: {self.test_metrics_csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="Run folder under logs/, e.g. v1-run3")
    parser.add_argument("--tag", default="eval", help="Suffix for the output CSV")
    parser.add_argument("--tune-thresholds", action="store_true",
                        help="Search per-channel thresholds on the val split first")
    parser.add_argument("--no-tta", action="store_true", help="Disable flip-TTA")
    parser.add_argument("--no-postprocess", action="store_true",
                        help="Disable connected-component cleanup")
    args = parser.parse_args()

    if args.no_tta:
        cfg.infer.tta_flips = False
    if args.no_postprocess:
        cfg.infer.min_component_voxels = (0, 0, 0)
        cfg.infer.min_total_voxels = (0, 0, 0)

    run_dir = os.path.join(cfg.paths.logs_dir, args.run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"No such run: {run_dir}")

    paths = _EvalPaths(run_dir, args.tag)
    if not os.path.exists(paths.checkpoint_path):
        raise SystemExit(f"No checkpoint at {paths.checkpoint_path}")

    device = get_device()
    model = build_model(cfg, device)
    loaders = build_dataloaders(cfg)

    if args.tune_thresholds:
        tune_and_save(
            model, loaders["val_loader"], device, cfg,
            paths.checkpoint_path, paths.eval_dir,
            inferer=lambda m, x: run_inference(m, x, cfg),
        )

    # run_test reloads the checkpoint itself, so the tuning pass above cannot
    # leave the model in a modified state.
    run_test(model, loaders, loss_fn=None, device=device, cfg=cfg, run_logger=paths)

    save_infer_config(cfg, os.path.join(paths.eval_dir, f"infer_config_{args.tag}.json"))


if __name__ == "__main__":
    main()
