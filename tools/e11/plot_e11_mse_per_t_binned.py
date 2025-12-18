#!/usr/bin/env python3
"""
Plot a binned "MSE vs timestep t" curve from sparse training logs:
keys like mse_per_t/mse_tXXXX.

This is *not* a formal recon metric—it's a quick look at whether the loss
weighting seems to change error across t.

Usage:
  python tools/e11/plot_e11_mse_per_t_binned.py \
    docs/assets/e11/e11a_data/a_loss.jsonl \
    docs/assets/e11/e11b_data/b_loss.jsonl \
    --names e11a-baseline e11b-minsnr-g5 \
    --bins 40 \
    --out docs/assets/e11/e11_plots/e11_mse_per_t_binned_ab.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


_PAT = re.compile(r"^mse_per_t/mse_t(\d+)$")


def load_jsonl_flat(path: str) -> List[dict]:
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


def collect_sparse_mse(recs: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    ts = []
    mses = []
    for r in recs:
        for k, v in r.items():
            m = _PAT.match(k)
            if m:
                ts.append(int(m.group(1)))
                mses.append(float(v))
    if len(ts) == 0:
        raise RuntimeError("No mse_per_t/mse_tXXXX keys found.")
    return np.asarray(ts, dtype=np.int32), np.asarray(mses, dtype=np.float64)


def binned_mean(ts: np.ndarray, mses: np.ndarray, bins: int, t_max: int) -> Tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0, t_max, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    y = np.full((bins,), np.nan, dtype=np.float64)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (ts >= lo) & (ts < hi)
        if mask.any():
            y[i] = float(mses[mask].mean())
    return centers, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("loss_paths", nargs="+")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--t_max", type=int, default=1000, help="max t for binning (default: 1000)")
    ap.add_argument("--yscale", choices=["linear", "log"], default="log")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.names) != len(args.loss_paths):
        raise SystemExit(f"--names count ({len(args.names)}) must match inputs ({len(args.loss_paths)})")

    plt.figure()
    for name, lp in zip(args.names, args.loss_paths):
        recs = load_jsonl_flat(lp)
        ts, mses = collect_sparse_mse(recs)
        x, y = binned_mean(ts, mses, bins=int(args.bins), t_max=int(args.t_max))
        plt.plot(x, y, marker="o", label=name)

    plt.xlabel("t (binned)")
    plt.ylabel("mean mse_per_t")
    plt.title("E11: binned MSE vs t (from sparse logs)")
    if args.yscale == "log":
        plt.yscale("log")
    plt.legend()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
