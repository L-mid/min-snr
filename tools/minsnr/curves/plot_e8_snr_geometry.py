"""
Plots snr's geometric shape.

Useage:

python tools/<script>.py \
  LOSS_A RESULTS_A \
  LOSS_B RESULTS_B \
  LOSS_C RESULTS_C \
  --names e8a e8b e8c \
  --out docs/assets/e8/<something>.png

  
Current:
python tools/minsnr/curves/plot_e8_snr_geometry.py \
  docs/assets/e8/e8a_data/loss.jsonl \
  docs/assets/e8/e8b_data/loss.jsonl \
  docs/assets/e8/e8c_data/loss.jsonl \
  --names e8a e8b e8c \
  --out docs/assets/e8/e8_plots/e8_snr_geometry.png

"""




import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


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


def extract_step(rec):
    return rec.get("_i") or rec.get("step") or rec.get("global_step")


def collect_snr_grad_curv(loss_path):
    records = load_jsonl(loss_path)
    snr = []
    grad = []
    curv = []

    for r in records:
        if "mins_snr/snr_mean" not in r:
            continue
        s = r["mins_snr/snr_mean"]
        g = r.get("train/grad_global_L2")
        c = r.get("curvature/hutch_trace_mean")
        if s is None:
            continue
        snr.append(s)
        grad.append(g)
        curv.append(c)

    return snr, grad, curv

def main():
    parser = argparse.ArgumentParser(
        description="Plot grad and curvature vs SNR for Min-SNR runs."
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
        help="Labels for experiments.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output PNG path.",
    )
    args = parser.parse_args()

    if len(args.loss_files) != len(args.names):
        raise ValueError("Need one --names entry per loss file.")

    fig, (ax_grad, ax_curv) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for loss_path, name in zip(args.loss_files, args.names):
        snr, grad, curv = collect_snr_grad_curv(loss_path)

        # Filter None values
        snr_g = [s for s, g in zip(snr, grad) if g is not None]
        grad_g = [g for g in grad if g is not None]

        snr_c = [s for s, c in zip(snr, curv) if c is not None]
        curv_c = [c for c in curv if c is not None]

        if snr_g:
            ax_grad.scatter(snr_g, grad_g, alpha=0.4, label=name)
        if snr_c:
            ax_curv.scatter(snr_c, curv_c, alpha=0.4, label=name)

    ax_grad.set_ylabel("grad_global_L2")
    ax_grad.set_title("Grad norm vs SNR (per step)")
    ax_grad.legend()
    ax_grad.grid(True)

    ax_curv.set_ylabel("Hutch trace")
    ax_curv.set_xlabel("mins_snr/snr_mean")
    ax_curv.set_title("Curvature vs SNR (per step)")
    ax_curv.legend()
    ax_curv.grid(True)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()


