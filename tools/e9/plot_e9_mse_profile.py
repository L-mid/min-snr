"""
Usage example (with e9 data):

python tools/e9/plot_e9_mse_profile.py \
  docs/assets/e9/e9a_data/loss.jsonl \
  docs/assets/e9/e9b_data/loss.jsonl \
  --names e9a-vanilla e9b-min-snr  \
  --out docs/assets/e9/e9_plots/e9_mse_profile_e9ab.png
 
Flags:
  --frac 0.3   # use last 30%% of steps (default) when averaging MSE(t)

Adjust paths/names for e9 accordingly.
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def load_mse_profile(path: Path, frac: float = 0.3):
    """
    Aggregate mse_per_t/mse_tXXXX over the last `frac` fraction of training steps.

    Returns:
        ts:  sorted list of timesteps (ints)
        mses: mean MSE per timestep (floats)
    """
    records = []
    with path.open("r") as f:
        for line in f:
            rec = json.loads(line)
            step = rec.get("_i")
            out = rec.get("out", {})
            records.append((step, out))

    if not records:
        return [], []

    max_step = max(step for step, _ in records if step is not None)
    cutoff = max_step * (1.0 - frac)

    mse_by_t = {}

    for step, out in records:
        if step is None or step < cutoff:
            continue

        for key, val in out.items():
            if not key.startswith("mse_per_t/"):
                continue
            # Example key: "mse_per_t/mse_t0231"
            idx = key.rfind("mse_t")
            if idx == -1:
                continue
            t_str = key[idx + 5 :]
            try:
                t = int(t_str)
            except ValueError:
                continue

            mse_by_t.setdefault(t, []).append(val)

    if not mse_by_t:
        return [], []

    ts = sorted(mse_by_t.keys())
    mses = [sum(mse_by_t[t]) / len(mse_by_t[t]) for t in ts]
    return ts, mses


def main():
    parser = argparse.ArgumentParser(
        description="Plot mean MSE(t) profile vs timestep from mse_per_t/mse_tXXXX logs."
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
    parser.add_argument(
        "--frac",
        type=float,
        default=0.3,
        help="Fraction of the *end* of training to average over (default 0.3 = last 30%%).",
    )
    args = parser.parse_args()

    if len(args.loss_files) != len(args.names):
        raise ValueError("Number of loss_files must match number of --names.")

    loss_files = [Path(p) for p in args.loss_files]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("timestep index t")
    ax.set_ylabel("log10 MSE(t) over last {:.0%} of training".format(args.frac))

    for path, name in zip(loss_files, args.names):
        ts, mses = load_mse_profile(path, frac=args.frac)
        if not ts:
            print(f"[WARN] No MSE(t) data found in {path}")
            continue

        # log10 for readability
        ys = [math.log10(m) if m > 0 else float("nan") for m in mses]

        ax.plot(ts, ys, label=name)

    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()