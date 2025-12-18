#!/usr/bin/env python3
"""
Usage:
  python tools/e12/plot_e12_curves_vs_images_seen.py \
    LOSS_A RESULTS_A \
    LOSS_B RESULTS_B \
    LOSS_C RESULTS_C \
    --names e12a e12b e12c \
    --batch_sizes 4 64 128 \
    --out docs/assets/e12/e12_plots/e12_curves_vs_images_seen_abc.png

Current:
python tools/e12/plot_e12_curves_vs_images_seen.py \
  docs/assets/e12/e12a_data/a_loss.jsonl docs/assets/e12/e12a_data/a_results.jsonl \
  docs/assets/e12/e12b_data/b_loss.jsonl docs/assets/e12/e12b_data/b_results.jsonl \
  docs/assets/e12/e12c_data/c_loss.jsonl docs/assets/e12/e12c_data/c_results.jsonl \
  --names e12a-bs4 e12b-bs64 e12c-bs128 \
  --batch_sizes 4 64 128 \
  --out docs/assets/e12/e12_plots/e12_curves_vs_images_seen_abc.png
        
    
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_jsonl(path: str):
    recs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if isinstance(r, dict) and isinstance(r.get("out"), dict):
                out = dict(r["out"])
                out["_i"] = r.get("_i")
                r = out
            recs.append(r)
    return recs


def extract_xy(loss_path: str, key: str, batch_size: int):
    recs = load_jsonl(loss_path)
    xs, ys = [], []
    for r in recs:
        if not isinstance(r, dict):
            continue
        step = r.get("_i", None)
        if step is None:
            continue
        if key in r:
            xs.append(int(step) * int(batch_size))
            ys.append(float(r[key]))
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="LOSS/RESULTS pairs (LOSS1 RES1 LOSS2 RES2 ...)")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--batch_sizes", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.paths) % 2 != 0:
        raise SystemExit("Expected even number of positional args: LOSS1 RES1 LOSS2 RES2 ...")
    pairs = [(args.paths[i], args.paths[i + 1]) for i in range(0, len(args.paths), 2)]
    if len(args.names) != len(pairs):
        raise SystemExit(f"--names must have {len(pairs)} entries.")
    if len(args.batch_sizes) != len(pairs):
        raise SystemExit(f"--batch_sizes must have {len(pairs)} entries.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    panels = [
        ("train/loss", axes[0, 0], "train/loss vs images_seen"),
        ("train/grad_global_L2", axes[0, 1], "grad_global_L2 vs images_seen"),
        ("val/fid", axes[1, 0], "FID vs images_seen"),
        ("val/kid", axes[1, 1], "KID vs images_seen"),
    ]

    for (loss_path, _res_path), name, bs in zip(pairs, args.names, args.batch_sizes):
        for key, ax, title in panels:
            xs, ys = extract_xy(loss_path, key=key, batch_size=bs)
            if xs:
                ax.plot(xs, ys, label=name)
            ax.set_title(title)
            ax.set_xlabel("images_seen")
            ax.set_ylabel(key)

    for ax in axes.flatten():
        ax.legend()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
