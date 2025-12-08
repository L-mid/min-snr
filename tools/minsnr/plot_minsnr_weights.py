"""
Plots Min-SNR weights.

Usage (two-run mode, if both have mins_snr_curve logged):

    python tools/minsnr/plot_minsnr_weights.py \
        docs/assets/e3/e3_data/loss.jsonl \
        docs/assets/e4/e4_data/loss.jsonl \
        --out docs/assets/e4/e4_plots/minsnr_weights_raw_vs_norm.png

Current (one-run mode, derive norm from raw):

    python tools/minsnr/plot_minsnr_weights.py \
        docs/assets/e3/e3_data/loss.jsonl \
        --out docs/assets/e4/e4_plots/minsnr_weights_raw_vs_norm_e3.png
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def load_curve(path):
    """Finds mins_snr_curve/t and mins_snr_curve/weight from jsonl."""
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            data = rec.get("out", rec)

            if "mins_snr_curve/t" in data and "mins_snr_curve/weight" in data:
                t = np.array(data["mins_snr_curve/t"], dtype=float)
                w = np.array(data["mins_snr_curve/weight"], dtype=float)
                return t, w

    raise RuntimeError(f"No mins_snr_curve found in {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("e_raw_loss", type=str)          # e.g. E2/E3 (raw Min-SNR)
    ap.add_argument("e_norm_loss", type=str, nargs="?", default=None)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    # Raw curve always comes from the first file
    t_raw, w_raw = load_curve(args.e_raw_loss)

    if args.e_norm_loss is None:
        # One-run mode: construct the mean-normalized curve from the same raw data.
        w_norm = w_raw / w_raw.mean()
        t_norm = t_raw
        norm_label = "Min-SNR (mean-normalized, derived)"
    else:
        # Two-run mode: load curve from the second run (e.g. E4)
        t_norm, w_norm = load_curve(args.e_norm_loss)
        norm_label = "Min-SNR (mean-normalized)"

    t_raw_norm = t_raw / t_raw.max()
    t_norm_norm = t_norm / t_norm.max()

    plt.figure()
    plt.plot(t_raw_norm, w_raw, label="raw Min-SNR")
    plt.plot(t_norm_norm, w_norm, linestyle="--", label=norm_label)
    plt.axhline(1.0, linestyle=":", label="scale = 1")

    plt.xlabel("t / T")
    plt.ylabel("loss weight w(t)")
    plt.title("Min-SNR weights: raw vs mean-normalized")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()