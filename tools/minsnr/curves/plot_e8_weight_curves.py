"""
Plots the min-snr weight curves.


Useage:

python tools/<script>.py \
  LOSS_A RESULTS_A \
  LOSS_B RESULTS_B \
  LOSS_C RESULTS_C \
  --names e8a e8b e8c \
  --out docs/assets/e8/<something>.png

  
Current:

python tools/minsnr/curves/plot_e8_weight_curves.py \
  docs/assets/e8/e8a_data/loss.jsonl \
  docs/assets/e8/e8b_data/loss.jsonl \
  docs/assets/e8/e8c_data/loss.jsonl \
  --names e8a e8b e8c \
  --out docs/assets/e8/e8_plots/e8_weight_curves.png

"""




import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

def load_jsonl(path):
    """This load jsonl correctly extracts."""
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
            t = r["mins_snr_curve/t"]
            w = r["mins_snr_curve/weight"]
            return t, w
    raise RuntimeError(f"No mins_snr_curve/t & mins_snr_curve/weight found in {loss_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Min-SNR weight curves w_gamma(t) vs t for multiple experiments."
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
        help="Labels for each experiment (e.g. e8a-gamma1, e8b-gamma3...).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output PNG path.",
    )
    args = parser.parse_args()

    if len(args.loss_files) != len(args.names):
        raise ValueError("Need one --names entry per loss file.")

    fig, ax = plt.subplots(figsize=(8, 5))

    for loss_path, name in zip(args.loss_files, args.names):
        t, w = find_minsnr_curve(loss_path)
        # Normalize t to [0,1] for nicer comparison
        t_norm = [ti / (max(t) if max(t) > 0 else 1.0) for ti in t]
        ax.plot(t_norm, w, label=name)

    ax.set_xlabel("t / T (normalized timestep)")
    ax.set_ylabel("Min-SNR weight w_gamma(t)")
    ax.set_title("Min-SNR weight curves vs normalized timestep")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()