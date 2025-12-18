#!/usr/bin/env python3
"""
Plot E11 KID over training steps from loss.jsonl logs.

Usage:
  python tools/e11/plot_e11_kid_vs_steps.py \
    docs/assets/e11/e11a_data/a_loss.jsonl \
    docs/assets/e11/e11b_data/b_loss.jsonl \
    --names e11a-baseline e11b-minsnr-g5 \
    --out docs/assets/e11/e11_plots/e11_kid_vs_steps_ab.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_jsonl_flat(path: str) -> List[dict]:
    recs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Handle {"_i": step, "out": {...}} style logs
            if isinstance(r, dict) and isinstance(r.get("out"), dict):
                out = dict(r["out"])
                out["_i"] = r.get("_i")
                r = out
            recs.append(r)
    return recs


def extract_series(recs: List[dict], key: str) -> Tuple[List[int], List[float]]:
    xs, ys = [], []
    for r in recs:
        if key in r:
            step = r.get("_i")
            if step is None:
                continue
            xs.append(int(step))
            ys.append(float(r[key]))
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("loss_paths", nargs="+", help="loss.jsonl paths (one per run)")
    ap.add_argument("--names", nargs="+", required=True, help="legend names, same count as loss_paths")
    ap.add_argument("--key", default="val/kid", help="metric key inside loss logs (default: val/kid)")
    ap.add_argument("--out", required=True, help="output .png path")
    args = ap.parse_args()

    if len(args.names) != len(args.loss_paths):
        raise SystemExit(f"--names count ({len(args.names)}) must match inputs ({len(args.loss_paths)})")

    plt.figure()
    for name, lp in zip(args.names, args.loss_paths):
        recs = load_jsonl_flat(lp)
        xs, ys = extract_series(recs, args.key)
        if len(xs) == 0:
            raise SystemExit(f"No '{args.key}' found in {lp}")
        # Sort by step (just in case)
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        plt.plot(xs, ys, marker="o", label=name)

    plt.xlabel("step")
    plt.ylabel(args.key)
    plt.title("E11: KID vs steps")
    plt.legend()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
