"""
Usage example (with e8 data):

python tools/e9/plot_e9_curvature_grad.py \
  docs/assets/e8/e8a_data/loss.jsonl \
  docs/assets/e8/e8b_data/loss.jsonl \
  docs/assets/e8/e8c_data/loss.jsonl \
  --names e8a e8b e8c \
  --out docs/assets/e9/e9_plots/e9_curvature_grad_e8abc.png
 
  
current:
python tools/e9/plot_e9_curvature_grad.py \
  docs/assets/e9/e9a_data/loss.jsonl \
  docs/assets/e9/e9b_data/loss.jsonl \
  --names e9a-vanilla e9b-min-snr \
  --out docs/assets/e9/e9_plots/e9_curvature_grad_e9ab.png


Adjust paths/names for e9 runs as needed.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_curvature_and_grad(path: Path):
    steps_curv = []
    curv_vals = []
    steps_grad = []
    grad_vals = []

    with path.open("r") as f:
        for line in f:
            rec = json.loads(line)
            step = rec.get("_i")
            out = rec.get("out", {})

            if "curvature/hutch_trace_mean" in out:
                steps_curv.append(step)
                curv_vals.append(out["curvature/hutch_trace_mean"])

            if "train/grad_global_L2" in out:
                steps_grad.append(step)
                grad_vals.append(out["train/grad_global_L2"])

    return steps_curv, curv_vals, steps_grad, grad_vals


def main():
    parser = argparse.ArgumentParser(
        description="Plot curvature (Hutchinson trace) and grad global L2 vs steps."
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

    fig, ax_curv = plt.subplots(figsize=(8, 5))
    ax_curv.set_xlabel("Training step")
    ax_curv.set_ylabel("curvature/hutch_trace_mean")

    # Plot curvature
    all_data = []
    for path, name in zip(loss_files, args.names):
        steps_curv, curv_vals, steps_grad, grad_vals = load_curvature_and_grad(path)
        all_data.append((steps_curv, curv_vals, steps_grad, grad_vals, name))

        paired_curv = sorted(zip(steps_curv, curv_vals), key=lambda x: x[0])
        if paired_curv:
            s, v = zip(*paired_curv)
            ax_curv.plot(s, v, label=f"{name} curvature")

    # Grad norm on secondary y-axis
    ax_grad = ax_curv.twinx()
    ax_grad.set_ylabel("train/grad_global_L2")

    for steps_curv, curv_vals, steps_grad, grad_vals, name in all_data:
        paired_grad = sorted(zip(steps_grad, grad_vals), key=lambda x: x[0])
        if not paired_grad:
            continue
        s, v = zip(*paired_grad)
        ax_grad.plot(s, v, linestyle="--", label=f"{name} grad L2")

    # Combined legend
    lines1, labels1 = ax_curv.get_legend_handles_labels()
    lines2, labels2 = ax_grad.get_legend_handles_labels()
    ax_curv.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax_curv.grid(alpha=0.3)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()