"""Side-by-side comparison of two finished runs, e.g. the ViT and the Mamba encoder.

    python tools/compare_runs.py v2-run3 v3-mamba-30ep
    python tools/compare_runs.py v2-run3 v3-mamba-30ep --matched-epoch 30

Reads only what the runs wrote under logs/<run>/ and writes
logs/compare_<a>_vs_<b>/:
  curves.png   per-epoch validation curves of both runs on shared axes
               (mean Dice, ET Dice, mean HD95, train loss): the "growth" view
  summary.md   the comparison table: model size and cost, validation at the
               matched epoch and at each run's best, test metrics with the HD95
               breakdown, XAI (modality ablation, MC-dropout retention,
               deletion AUCs, randomisation check)
  summary.csv  the same table, machine-readable
Anything a run has not produced yet (still training, a component not run) is
reported as n/a, never estimated. Parameter counts come from rebuilding each
run's model from its config_snapshot.json on the meta device (no weights).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHANNELS = ("tc", "wt", "et")
# Categorical slots 1 and 2 of the validated reference palette (dataviz skill):
# the run order on the command line fixes the color, never the ranking.
SERIES = ("#2a78d6", "#eb6834")
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


# ---------------------------------------------------------------------------
# reading a run
# ---------------------------------------------------------------------------

def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        parsed = {}
        for key, value in row.items():
            try:
                parsed[key] = float(value)
            except (TypeError, ValueError):
                parsed[key] = value
        out.append(parsed)
    return out


def _param_counts(snapshot):
    """(total, encoder) parameter counts, rebuilt from the snapshot on meta."""
    try:
        import torch
        from config import cfg
        from models.unetr import UNETR
        for key, value in (snapshot or {}).get("unetr", {}).items():
            cfg.unetr[key] = tuple(value) if key == "img_shape" else value
        encoder = (snapshot or {}).get("unetr", {}).get("encoder", "vit")
        mamba = dict(cfg.mamba)
        mamba.update((snapshot or {}).get("mamba", {}))
        with torch.device("meta"):
            model = UNETR(img_shape=tuple(cfg.unetr.img_shape), input_dim=cfg.unetr.input_dim,
                          output_dim=cfg.unetr.output_dim, embed_dim=cfg.unetr.embed_dim,
                          patch_size=cfg.unetr.patch_size, num_heads=cfg.unetr.num_heads,
                          dropout=cfg.unetr.dropout, encoder=encoder,
                          mamba_kwargs=mamba if encoder == "mamba" else None)
        total = sum(p.numel() for p in model.parameters())
        enc = sum(p.numel() for name in model.encoder_module_names()
                  for p in getattr(model, name).parameters())
        return total, enc
    except Exception as exc:                                        # noqa: BLE001
        print(f"[compare] could not rebuild the model for a param count: {exc}")
        return None, None


def load_run(logs_dir, name):
    run_dir = os.path.join(logs_dir, name)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"No such run: {run_dir}")
    snapshot = _read_json(os.path.join(run_dir, "config_snapshot.json")) or {}
    test = _read_csv(os.path.join(run_dir, "testing", "test_metrics.csv"))
    run = {
        "name": name,
        "encoder": snapshot.get("unetr", {}).get("encoder", "vit"),
        "epochs_planned": snapshot.get("epoch"),
        "metrics": _read_csv(os.path.join(run_dir, "metrics.csv")),
        "test": test[0] if test else {},
        "sweep": _read_json(os.path.join(run_dir, "eval", "threshold_sweep.json")) or {},
        "modality": _read_json(os.path.join(run_dir, "xai", "modality.json")),
        "uncertainty": _read_json(os.path.join(run_dir, "xai", "uncertainty.json")),
        "faithful": _read_json(os.path.join(run_dir, "xai", "faithful.json")),
    }
    run["params"], run["encoder_params"] = _param_counts(snapshot)
    run["label"] = f"{'Mamba' if run['encoder'] == 'mamba' else 'ViT'} ({name})"
    return run


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

def _row_at(metrics, epoch):
    for row in metrics:
        if int(row["epoch"]) == epoch:
            return row
    return None


def _best(metrics, upto=None):
    rows = [r for r in metrics if upto is None or int(r["epoch"]) <= upto]
    return max(rows, key=lambda r: r["mean_dice"]) if rows else None


def _retention(uncertainty, channel, fraction=0.02):
    if not uncertainty or "error" in uncertainty:
        return None
    rows = [v for k, v in uncertainty.items() if k.endswith(channel.upper())]
    if not rows or fraction not in rows[0]["retention_fractions"]:
        return None
    i = rows[0]["retention_fractions"].index(fraction)
    return (float(np.mean([r["retention_curve"][0] for r in rows])),
            float(np.mean([r["retention_curve"][i] for r in rows])))


def _deletion_auc(faithful, method):
    if not faithful or "error" in faithful:
        return None
    values = [v[method]["deletion_auc"] for k, v in faithful.items()
              if k != "sanity_check" and method in v]
    return float(np.mean(values)) if values else None


def build_rows(a, b, matched):
    """(section, metric, value_a, value_b, direction) with direction 'up'
    (higher is better), 'down' (lower is better) or None (descriptive)."""
    rows = []

    def add(section, metric, fn, direction=None):
        rows.append((section, metric, fn(a), fn(b), direction))

    def mcol(row, key):
        return None if row is None else row.get(key)

    add("model", "encoder", lambda r: r["encoder"])
    add("model", "parameters, total (M)",
        lambda r: None if r["params"] is None else r["params"] / 1e6)
    add("model", "parameters, encoder (M)",
        lambda r: None if r["encoder_params"] is None else r["encoder_params"] / 1e6)
    add("model", "epochs trained", lambda r: len(r["metrics"]) or None)
    add("model", "min / epoch (mean)",
        lambda r: np.mean([m["epoch_time_sec"] for m in r["metrics"]]) / 60 if r["metrics"] else None,
        "down")
    add("model", "training hours",
        lambda r: sum(m["epoch_time_sec"] for m in r["metrics"]) / 3600 if r["metrics"] else None,
        "down")

    for key, label, direction in (
            ("mean_dice", "mean Dice", "up"), ("dice_tc", "Dice TC", "up"),
            ("dice_wt", "Dice WT", "up"), ("dice_et", "Dice ET", "up"),
            ("sens_et", "sensitivity ET", "up"), ("mean_hd95", "mean HD95 (mm)", "down"),
            ("hd95_et", "HD95 ET (mm)", "down")):
        add(f"val @ epoch {matched}", label,
            lambda r, k=key: mcol(_row_at(r["metrics"], matched), k), direction)
    add(f"val, best within {matched} ep", "mean Dice",
        lambda r: mcol(_best(r["metrics"], matched), "mean_dice"), "up")
    add(f"val, best within {matched} ep", "at epoch",
        lambda r: mcol(_best(r["metrics"], matched), "epoch"))
    add("val, best overall", "mean Dice", lambda r: mcol(_best(r["metrics"]), "mean_dice"), "up")
    add("val, best overall", "at epoch", lambda r: mcol(_best(r["metrics"]), "epoch"))

    for key, label, direction in (
            ("mean_dice", "mean Dice", "up"), ("dice_tc", "Dice TC", "up"),
            ("dice_wt", "Dice WT", "up"), ("dice_et", "Dice ET", "up"),
            ("mean_hd95", "mean HD95 (mm)", "down"), ("hd95_tc", "HD95 TC (mm)", "down"),
            ("hd95_wt", "HD95 WT (mm)", "down"), ("hd95_et", "HD95 ET (mm)", "down"),
            ("hd95_et_clean", "HD95 ET, both-non-empty cases (mm)", "down"),
            ("n_halluc_et", "ET hallucinated (of n_test)", "down"),
            ("n_miss_et", "ET missed (of n_test)", "down"),
            ("sens_tc", "sensitivity TC", "up"), ("sens_wt", "sensitivity WT", "up"),
            ("sens_et", "sensitivity ET", "up"), ("n_test", "n_test", None)):
        add("test (TTA + tuned thresholds + post-proc)", label,
            lambda r, k=key: r["test"].get(k), direction)
    add("test (TTA + tuned thresholds + post-proc)", "tuned thresholds TC/WT/ET",
        lambda r: "/".join(f"{t:g}" for t in r["sweep"]["best"]) if r["sweep"].get("best") else None)

    def modality(r, ablated, channel):
        m = r["modality"]
        if not m or "error" in m:
            return None
        return m["delta"][ablated][CHANNELS.index(channel)]

    for ablated in ("FLAIR", "T1", "T1ce", "T2"):
        for channel in CHANNELS:
            add("XAI: Dice change when a modality is removed", f"-{ablated} -> {channel.upper()}",
                lambda r, a_=ablated, c=channel: modality(r, a_, c))
    for channel in CHANNELS:
        add("XAI: MC-dropout, Dice after referring 2% most-uncertain voxels",
            f"{channel.upper()} (before -> after)",
            lambda r, c=channel: None if _retention(r["uncertainty"], c) is None
            else "{:.3f} -> {:.3f}".format(*_retention(r["uncertainty"], c)))
    add("XAI: MC-dropout, Dice after referring 2% most-uncertain voxels", "dropout layers sampled",
        lambda r: None if not r["uncertainty"] or "error" in r["uncertainty"]
        else next(iter(r["uncertainty"].values()))["n_dropout_layers"])
    for method, label in (("hires", "HiResCAM"), ("grad", "Grad-CAM"),
                          ("rollout", "attention / hidden-attention rollout"),
                          ("random", "random ordering (null)")):
        add("XAI: deletion AUC (lower = more faithful)", label,
            lambda r, m=method: _deletion_auc(r["faithful"], m), "down")
    add("XAI: deletion AUC (lower = more faithful)", "randomisation SSIM, fully randomised",
        lambda r: None if not r["faithful"] or "sanity_check" not in r["faithful"]
        else r["faithful"]["sanity_check"]["cascade"][-1]["ssim_vs_trained"], "down")
    return rows


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if float(value).is_integer() and abs(value) >= 1:
        return f"{int(value)}"
    return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"


def _better(a, b, direction, name_a, name_b):
    if direction is None or not all(isinstance(v, (int, float)) for v in (a, b)) or a == b:
        return ""
    return name_b if (b > a) == (direction == "up") else name_a


def write_tables(rows, a, b, out_dir, matched):
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["section", "metric", a["name"], b["name"], "difference (b - a)", "better"])
        for section, metric, va, vb, direction in rows:
            diff = vb - va if all(isinstance(v, (int, float)) for v in (va, vb)) else ""
            writer.writerow([section, metric, _fmt(va), _fmt(vb),
                             _fmt(diff) if diff != "" else "",
                             _better(va, vb, direction, a["name"], b["name"])])

    lines = [f"# {a['label']} vs {b['label']}", "",
             f"Matched epoch: {matched}. {a['name']} trained {len(a['metrics'])} epochs, "
             f"{b['name']} trained {len(b['metrics'])}; each with its own warmup + cosine "
             f"schedule over its own epoch budget, so the matched-epoch rows compare the "
             f"runs at the same number of epochs, not at the same point of their schedules.",
             "", "Test split is n=70: an ET Dice difference below ~0.03 is inside its noise.",
             "\"better\" follows each metric's direction only; it is not a significance test.", ""]
    section = None
    for sec, metric, va, vb, direction in rows:
        if sec != section:
            section = sec
            lines += ["", f"## {sec}", "",
                      f"| metric | {a['name']} | {b['name']} | b - a | better |",
                      "|---|---|---|---|---|"]
        diff = vb - va if all(isinstance(v, (int, float)) for v in (va, vb)) else None
        lines.append(f"| {metric} | {_fmt(va)} | {_fmt(vb)} | "
                     f"{_fmt(diff) if diff is not None else ''} | "
                     f"{_better(va, vb, direction, a['name'], b['name'])} |")
    md_path = os.path.join(out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return md_path, csv_path


# ---------------------------------------------------------------------------
# the curves
# ---------------------------------------------------------------------------

def plot_curves(a, b, out_dir, matched):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    panels = (("mean_dice", "Validation mean Dice", "higher is better", None),
              ("dice_et", "Validation ET Dice", "higher is better", None),
              ("mean_hd95", "Validation mean HD95 (mm)", "lower is better", 5),
              ("train_loss", "Training loss", "lower is better", None))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.93, top=0.86, bottom=0.08, hspace=0.42, wspace=0.28)

    for ax, (key, title, note, skip_first) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        ends = []
        for run, color in zip((a, b), SERIES):
            pts = [(int(r["epoch"]), r[key]) for r in run["metrics"]
                   if isinstance(r.get(key), float)]
            if not pts:
                continue
            x, y = zip(*pts)
            ax.plot(x, y, color=color, linewidth=1.6, solid_joinstyle="round",
                    solid_capstyle="round", label=run["label"], zorder=3)
            ax.plot(x[-1], y[-1], "o", color=color, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
            ends.append((x[-1], y[-1], run))
        # HD95 is dominated by the 374mm empty-mask sentinel in the first
        # epochs; scale the axis to the settled part and say so.
        if skip_first:
            settled = [r[key] for run in (a, b) for r in run["metrics"]
                       if int(r["epoch"]) > skip_first and isinstance(r.get(key), float)]
            if settled:
                ax.set_ylim(min(settled) * 0.9, max(settled) * 1.1)
                note += f"; epochs 1-{skip_first} off-scale"
        ax.axvline(matched, color=MUTED, linewidth=0.8, zorder=1)
        note += f"; vertical line = epoch {matched}"
        ax.set_title(title, loc="left", color=INK, fontsize=10.5, fontweight="bold", pad=16)
        ax.text(0, 1.02, note, transform=ax.transAxes, color=INK_2, fontsize=8, va="bottom")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, length=0)
        ax.set_xlabel("epoch", color=INK_2)
        # Direct end labels only when the two ends do not collide.
        if len(ends) == 2:
            lo, hi = ax.get_ylim()
            if abs(ends[0][1] - ends[1][1]) > 0.06 * (hi - lo):
                for x_end, y_end, run in ends:
                    ax.annotate(f"{run['label'].split(' ')[0]} {y_end:.3g}", (x_end, y_end),
                                xytext=(6, 0), textcoords="offset points", va="center",
                                color=INK_2, fontsize=8)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.07, 0.965), ncol=2,
               frameon=False, labelcolor=INK, fontsize=9.5)
    fig.text(0.07, 0.985, f"{a['label']} vs {b['label']}: validation per epoch",
             color=INK, fontsize=12, fontweight="bold", va="top")
    path = os.path.join(out_dir, "curves.png")
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", help="first run under logs/ (e.g. the ViT, v2-run3)")
    parser.add_argument("run_b", help="second run under logs/ (e.g. the Mamba run)")
    parser.add_argument("--matched-epoch", type=int, default=None,
                        help="epoch to compare at (default: the shorter run's length)")
    parser.add_argument("--logs-dir", default=os.path.join(ROOT, "logs"))
    args = parser.parse_args()

    a, b = load_run(args.logs_dir, args.run_a), load_run(args.logs_dir, args.run_b)
    lengths = [len(r["metrics"]) for r in (a, b) if r["metrics"]]
    matched = args.matched_epoch or (min(lengths) if lengths else 1)
    out_dir = os.path.join(args.logs_dir, f"compare_{a['name']}_vs_{b['name']}")
    os.makedirs(out_dir, exist_ok=True)

    md_path, csv_path = write_tables(build_rows(a, b, matched), a, b, out_dir, matched)
    png_path = plot_curves(a, b, out_dir, matched)
    print(open(md_path).read())
    print(f"wrote {md_path}\nwrote {csv_path}\nwrote {png_path}")


if __name__ == "__main__":
    main()
