"""
Plots mse loss per t.


python tools/plot_per_t_mse_profile.py \
    docs/assets/e3/e3_data/loss.jsonl \
  docs/assets/e4/e4_data/loss.jsonl \
  --names e3-snr-10k e4-minsnr-norm-10k \
  --out docs/assets/e4/e4_plots/per_t_mse_profile_e1e2e4.png


"""


import argparse
import json
import re

import matplotlib.pyplot as plt
import numpy as np

MSE_PREFIX = "mse_per_t/mse_t"


def extract_last_profile(path):
    """
    Returns (t, mse) from the last record that has any mse_per_t/... keys.
    """
    last = None
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            data = rec.get("out", rec)
            has_mse_per_t = any(k.startswith(MSE_PREFIX) for k in data.keys())
            if has_mse_per_t:
                last = data

    if last is None:
        raise RuntimeError(f"No mse_per_t entries found in {path}")

    t_vals = []
    mse_vals = []
    for k, v in last.items():
        if not k.startswith(MSE_PREFIX):
            continue
        # key format: mse_per_t/mse_t0004
        m = re.search(r"mse_t(\d+)", k)
        if m is None:
            continue
        t_idx = int(m.group(1))
        t_vals.append(t_idx)
        mse_vals.append(float(v))

    order = np.argsort(t_vals)
    t = np.array(t_vals)[order]
    mse = np.array(mse_vals)[order]

    print("OK.")
    return t, mse




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("loss_files", nargs="+", type=str)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    assert len(args.loss_files) == len(args.names)

    plt.figure()
    for path, name in zip(args.loss_files, args.names):
        t, mse = extract_last_profile(path)
        plt.plot(t / t.max(), mse, label=name)  # normalize t to [0,1]

    plt.xlabel("t / T")
    plt.ylabel("ε-MSE(t) (approx, last batch)")
    plt.title("Per-t epsilon MSE profile (late in training)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()