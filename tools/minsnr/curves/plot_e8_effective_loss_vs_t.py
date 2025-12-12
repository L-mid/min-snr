"""
Plots effective loss vs t.


Useage:

python tools/<script>.py \
  LOSS_A RESULTS_A \
  LOSS_B RESULTS_B \
  LOSS_C RESULTS_C \
  --names e8a e8b e8c \
  --out docs/assets/e8/<something>.png

  
Current:

python tools/minsnr/curves/plot_e8_effective_loss_vs_t.py \
  docs/assets/e8/e8a_data/loss.jsonl \
  docs/assets/e8/e8b_data/loss.jsonl \
  docs/assets/e8/e8c_data/loss.jsonl \
  --names e8a e8b e8c \
  --out docs/assets/e8/e8_plots/effective_loss_vs_t.png


"""



import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

MSE_PREFIX = "mse_per_t/mse_t"


def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Handle {"_i": step, "out": {...}} style logs
            if isinstance(r, dict) and isinstance(r.get("out"), dict):
                out = r["out"]
                out["_i"] = r.get("_i")  # optional: keep step around
                r = out
            records.append(r)
    return records


def find_minsnr_curve(loss_path):
    records = load_jsonl(loss_path)
    for r in records:
        if "mins_snr_curve/t" in r and "mins_snr_curve/weight" in r:
            return r["mins_snr_curve/t"], r["mins_snr_curve/weight"]
    raise RuntimeError(f"No mins_snr_curve found in {loss_path}")


def aggregate_mse_per_t(loss_path):
    records = load_jsonl(loss_path)
    sums = defaultdict(float)
    counts = defaultdict(int)

    for r in records:
        for k, v in r.items():
            if k.startswith(MSE_PREFIX):
                # k like "mse_per_t/mse_t123"
                m = re.search(r"mse_t(\d+)$", k)
                if not m:
                    continue
                t_idx = int(m.group(1))
                if isinstance(v, (int, float)):
                    sums[t_idx] += float(v)
                    counts[t_idx] += 1

    ts = sorted(sums.keys())
    mse = [sums[t] / counts[t] for t in ts]
    return ts, mse


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-t MSE(t) and w_gamma(t)*MSE(t) vs normalized t."
    )
    parser.add_argument(
        "loss_files",
        nargs="+",
        help="loss.jsonl files (one per experiment).",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Labels for each experiment.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output PNG path.",
    )
    args = parser.parse_args()

    if len(args.loss_files) != len(args.names):
        raise ValueError("Need one --names entry per loss file.")

    fig, (ax_mse, ax_eff) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for loss_path, name in zip(args.loss_files, args.names):
        t_curve, w_curve = find_minsnr_curve(loss_path)
        ts_mse, mse = aggregate_mse_per_t(loss_path)

        if not ts_mse:
            print(f"Warning: no mse_per_t entries in {loss_path}, skipping.")
            continue

        max_t = max(max(t_curve), max(ts_mse))
        t_norm_mse = [t / max_t for t in ts_mse]

        # Interpolate / align weight curve onto mse t's (simple index match if same)
        # Here we assume t indices line up; if not, we clamp / index with min(len(w_curve)).
        eff = []
        for t, mse_t in zip(ts_mse, mse):
            idx = min(t, len(w_curve) - 1)
            eff.append(w_curve[idx] * mse_t)

        # Plot MSE(t)
        ax_mse.plot(t_norm_mse, mse, label=name)
        # Plot effective w(t)*MSE(t)
        ax_eff.plot(t_norm_mse, eff, label=name)

    ax_mse.set_ylabel("MSE(t)")
    ax_mse.set_title("Unweighted per-t MSE(t)")
    ax_mse.legend()
    ax_mse.grid(True)

    ax_eff.set_ylabel("w_gamma(t) * MSE(t)")
    ax_eff.set_xlabel("t / T (normalized timestep)")
    ax_eff.set_title("Effective loss contribution vs t")
    ax_eff.legend()
    ax_eff.grid(True)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()