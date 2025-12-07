"""
Grabs the grad_global_L2 I calculated.

Useage:
    python tools/plot_grad_stats.py \
    docs/assets/e3/e3_data/loss.jsonl \
    docs/assets/e4/e4_data/loss.jsonl \
    --names e3-minsnr-short e4-minsnr-norm \
    --out docs/assets/e4/e4_plots/grad_global_L2_e3e4.png    


"""


import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def load_grad_series(path, value_keys):
    """Get stats from jsonl."""
    steps = []
    series = {k: [] for k in value_keys}

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out = rec.get("out", rec)

            # step: prefer explicit step/global_step, else use top-level _i
            step = out.get("step", out.get("global_step", rec.get("_i", None)))
            if step is None:
                continue

            # only keep records that actually have at least one grad key
            has_any = False
            for k in value_keys:
                if k in out:
                    has_any = True
                    break
            if not has_any:
                continue

            steps.append(int(step))
            for k in value_keys:
                val = out.get(k, float("nan"))
                series[k].append(float(val))

    if not steps:
        raise RuntimeError(f"No grad stats found in {path}")

    steps = np.array(steps)
    series = {k: np.array(v) for k, v in series.items()}
    return steps, series


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("loss_files", nargs="+", type=str)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    assert len(args.loss_files) == len(args.names)

    value_keys = [
        "train/grad_global_L2",
        # optionally add these if you want extra curves:
        # "train/grad_abs_mean",
        # "train/grad_abs_max",
    ]

    plt.figure()
    for path, name in zip(args.loss_files, args.names):
        try:
            steps, series = load_grad_series(path, value_keys)
        except RuntimeError:
            continue

        # primary curve: global L2
        if "train/grad_global_L2" in series:
            plt.plot(steps, series["train/grad_global_L2"], label=name)

    plt.xlabel("training step (from _i)")
    plt.ylabel("train/grad_global_L2")
    plt.title("Gradient behaviour over training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("OK.")

if __name__ == "__main__":
    main()