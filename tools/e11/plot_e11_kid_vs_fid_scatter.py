#!/usr/bin/env python3
"""
Plot E11 final KID vs final FID per run (scatter).

Usage:
  python tools/e11/plot_e11_kid_vs_fid_scatter.py \
    docs/assets/e11/e11a_data/a_loss.jsonl docs/assets/e11/e11a_data/a_results.jsonl \
    docs/assets/e11/e11b_data/b_loss.jsonl docs/assets/e11/e11b_data/b_results.jsonl \
    --names e11a-baseline e11b-minsnr-g5 \
    --out docs/assets/e11/e11_plots/e11_kid_vs_fid_ab.png
"""
from __future__ import annotations
 
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[dict]:
    recs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            recs.append(json.loads(line))
    return recs


def load_loss_flat(path: str) -> List[dict]:
    recs = []
    for r in load_jsonl(path):
        if isinstance(r, dict) and isinstance(r.get("out"), dict):
            out = dict(r["out"])
            out["_i"] = r.get("_i")
            r = out
        recs.append(r)
    return recs


def last_value(recs: List[dict], key: str) -> float:
    val = None
    for r in recs:
        if key in r:
            val = float(r[key])
    if val is None:
        raise KeyError(key)
    return val


def fid_from_results(results_path: str, key: str = "val/fid") -> float:
    recs = load_jsonl(results_path)
    if len(recs) == 0:
        raise RuntimeError(f"Empty results: {results_path}")
    r0 = recs[-1]
    if isinstance(r0, dict) and isinstance(r0.get("out"), dict):
        r0 = r0["out"]
    if key not in r0:
        raise KeyError(f"{key} not in {results_path}")
    return float(r0[key])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="pairs: LOSS RESULTS LOSS RESULTS ...")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--kid_key", default="val/kid")
    ap.add_argument("--fid_key", default="val/fid")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.paths) % 2 != 0:
        raise SystemExit("Expected an even number of positional args: LOSS RESULTS pairs.")
    n = len(args.paths) // 2
    if len(args.names) != n:
        raise SystemExit(f"--names count ({len(args.names)}) must match pair count ({n}).")

    kid_vals = []
    fid_vals = []

    for i in range(n):
        loss_p = args.paths[2 * i]
        res_p = args.paths[2 * i + 1]
        loss_recs = load_loss_flat(loss_p)
        kid = last_value(loss_recs, args.kid_key)
        fid = fid_from_results(res_p, args.fid_key)
        kid_vals.append(kid)
        fid_vals.append(fid)

    # Plot
    plt.figure()
    plt.scatter(kid_vals, fid_vals)

    # annotate
    for name, x, y in zip(args.names, kid_vals, fid_vals):
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4))

    plt.xlabel("KID (final)")
    plt.ylabel("FID (final)")
    plt.title("E11: KID vs FID (final)")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)

    # Print a tiny summary for the logs
    print("run,kid,fid")
    for name, k, f in zip(args.names, kid_vals, fid_vals):
        print(f"{name},{k:.6g},{f:.6g}")
    print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
