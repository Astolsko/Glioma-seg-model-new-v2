"""Redraw a run's training curves from its CSVs. No retraining, no GPU.

    python tools/replot.py --run v3-mamba-30ep
    python tools/replot.py --run <run> --font-size 16 --fig-size 8 5 --dpi 300 --format png pdf
    python tools/replot.py --run <run> --epoch-smoothing 0 --out figures/<run>

Reads logs/<run>/metrics.csv, plus train_steps.csv when the run has one (runs
from before 2026-09-14 do not, so they get no per-step or main-head train-loss
curve). Every flag defaults to its cfg.plot value in config.py. Writes loss /
dice / iou / hd95 / lr (/ loss_steps) into logs/<run>/plots/, replacing what is
there, unless --out is given.

Smoothing happens at plot time only: the CSVs are never modified, each raw curve
is drawn faintly under its smoothed one, and the weight is printed on the
figure. --epoch-smoothing 0 --step-smoothing 0 draws the logged values as they are.

A run trained with cfg.data.leak set (read from its config_snapshot.json) gets
the LEAKED SPLIT watermark on every figure, exactly as train.py drew it.
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import cfg
from utils.plot import leak_watermark, plot_metrics_from_csv


def _run_leak(run_dir):
    """cfg.data.leak as the run was trained, from its config_snapshot.json."""
    try:
        with open(os.path.join(run_dir, "config_snapshot.json")) as f:
            return (json.load(f).get("data") or {}).get("leak")
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="run folder under logs/, e.g. v3-mamba-30ep")
    parser.add_argument("--logs-dir", default=os.path.join(ROOT, cfg.paths.logs_dir))
    parser.add_argument("--out", default=None, help="output folder (default: logs/<run>/plots)")
    parser.add_argument("--font-family", default=cfg.plot.font_family)
    parser.add_argument("--font-size", type=float, default=cfg.plot.font_size)
    parser.add_argument("--fig-size", type=float, nargs=2, metavar=("W", "H"),
                        default=list(cfg.plot.fig_size), help="figure size in inches")
    parser.add_argument("--dpi", type=int, default=cfg.plot.dpi)
    parser.add_argument("--format", dest="formats", nargs="+", choices=("png", "pdf", "svg"),
                        default=list(cfg.plot.formats))
    parser.add_argument("--epoch-smoothing", type=float, default=cfg.plot.epoch_smoothing,
                        help="EMA weight in [0, 1) for the per-epoch curves; 0 = raw")
    parser.add_argument("--step-smoothing", type=float, default=cfg.plot.step_smoothing,
                        help="EMA weight in [0, 1) for the per-step loss curve; 0 = raw")
    args = parser.parse_args()

    run_dir = os.path.join(args.logs_dir, args.run)
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        raise SystemExit(f"No metrics.csv in {run_dir}")

    plot_metrics_from_csv(
        metrics_csv, args.out or os.path.join(run_dir, "plots"),
        train_steps_csv_path=os.path.join(run_dir, "train_steps.csv"),
        style={
            "font_family": args.font_family, "font_size": args.font_size,
            "fig_size": tuple(args.fig_size), "dpi": args.dpi, "formats": args.formats,
            "epoch_smoothing": args.epoch_smoothing, "step_smoothing": args.step_smoothing,
            "watermark": leak_watermark(_run_leak(run_dir)),
        },
    )


if __name__ == "__main__":
    main()
