"""
Usage example (with e8 data):

python tools/e9/plot_e9_fid_vs_loss.py \
  docs/assets/e9/e9a_data/loss.jsonl \
  --names e9a \
  --out docs/assets/e9/e9_plots/e9_fid_vs_loss_e8abc.png


"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_loss_and_fid(path: Path):
    steps_loss = []
    loss_vals = []
    steps_fid = []
    fid_vals = []

    with path.open("r") as f:
        for line in f:
            rec = json.loads(line)
            step = rec.get("_i")
            out = rec.get("out", {})

            # Training loss at every train log
            if "train/loss" in out:
                steps_loss.append(step)
                loss_vals.append(out["train/loss"])

            # FID only at eval milestones
            if "val/fid" in out:
                steps_fid.append(step)
                fid_vals.append(out["val/fid"])

    return steps_loss, loss_vals, steps_fid, fid_vals


def main():
    parser = argparse.ArgumentParser(
        description="Plot FID vs steps and train loss vs steps from loss.jsonl files."
    )
    parser.add_argument(
        "loss_files",
        nargs="+",
        help="One or more loss.jsonl files (one per run).",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Legend names for each run (must match number of loss files).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output PNG path.",
    )
    args = parser.parse_args()

    if len(args.loss_files) != len(args.names):
        raise ValueError("Number of loss_files must match number of --names.")

    loss_files = [Path(p) for p in args.loss_files]

    fig, ax_loss = plt.subplots(figsize=(8, 5))

    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("train/loss")

    # Plot loss curves
    all_data = []
    for path, name in zip(loss_files, args.names):
        steps_loss, loss_vals, steps_fid, fid_vals = load_loss_and_fid(path)
        all_data.append((steps_loss, loss_vals, steps_fid, fid_vals, name))

        # Sort by step just in case
        paired = sorted(zip(steps_loss, loss_vals), key=lambda x: x[0])
        if paired:
            s, v = zip(*paired)
            ax_loss.plot(s, v, label=f"{name} train/loss")

    # FID on secondary y-axis
    ax_fid = ax_loss.twinx()
    ax_fid.set_ylabel("val/fid")

    for steps_loss, loss_vals, steps_fid, fid_vals, name in all_data:
        paired_fid = sorted(zip(steps_fid, fid_vals), key=lambda x: x[0])
        if not paired_fid:
            continue
        s, v = zip(*paired_fid)
        # Markers at eval milestones
        ax_fid.plot(s, v, marker="o", linestyle="--", label=f"{name} val/fid")

    # Combined legend
    lines1, labels1 = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax_fid.get_legend_handles_labels()
    ax_loss.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax_loss.grid(alpha=0.3)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()