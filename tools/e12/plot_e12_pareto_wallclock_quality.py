#!/usr/bin/env python3
"""
Usage:
  python tools/e12/plot_e12_pareto_wallclock_quality.py \
    LOSS_A RESULTS_A \
    LOSS_B RESULTS_B \
    LOSS_C RESULTS_C \
    --names e12a e12b e12c \
    --reduce min \
    --out docs/assets/e12/e12_plots/e12_pareto_wallclock_quality_abc.png

current:
python tools/e12/plot_e12_pareto_wallclock_quality.py \
  docs/assets/e12/e12a_data/a_loss.jsonl docs/assets/e12/e12a_data/a_results.jsonl \
  docs/assets/e12/e12b_data/b_loss.jsonl docs/assets/e12/e12b_data/b_results.jsonl \
  docs/assets/e12/e12c_data/c_loss.jsonl docs/assets/e12/e12c_data/c_results.jsonl \
  --names e12a-bs4 e12b-bs64 e12c-bs128 \
  --reduce min \
  --out docs/assets/e12/e12_plots/e12_pareto_wallclock_quality_abc.png       
    
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


def load_single_results(results_path: str) -> dict:
    recs = load_jsonl(results_path)
    for r in reversed(recs):
        if isinstance(r, dict):
            return r
    return {}


def extract_series(loss_path: str, key: str):
    recs = load_jsonl(loss_path)
    xs = []
    for r in recs:
        if isinstance(r, dict) and key in r:
            xs.append(r[key])
    return xs


def reduce_series(xs, mode: str):
    if not xs:
        return None
    xs = [float(x) for x in xs]
    if mode == "last":
        return xs[-1]
    if mode == "min":
        return min(xs)
    if mode == "max":
        return max(xs)
    if mode == "mean":
        return sum(xs) / len(xs)
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="LOSS/RESULTS pairs (LOSS1 RES1 LOSS2 RES2 ...)")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--reduce", default="min", choices=["min", "last", "max", "mean"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.paths) % 2 != 0:
        raise SystemExit("Expected even number of positional args: LOSS1 RES1 LOSS2 RES2 ...")
    pairs = [(args.paths[i], args.paths[i + 1]) for i in range(0, len(args.paths), 2)]
    if len(args.names) != len(pairs):
        raise SystemExit(f"--names must have {len(pairs)} entries.")

    wall = []
    fid_y = []
    kid_y = []

    for (loss_path, res_path), name in zip(pairs, args.names):
        R = load_single_results(res_path)
        rt = float(R.get("run_time_s", R.get("_elapsed_sec", 0.0)))

        fid = reduce_series(extract_series(loss_path, "val/fid"), args.reduce)
        kid = reduce_series(extract_series(loss_path, "val/kid"), args.reduce)

        wall.append(rt)
        fid_y.append(fid)
        kid_y.append(kid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # FID vs wallclock
    ax = axes[0]
    for rt, y, name in zip(wall, fid_y, args.names):
        if y is None:
            continue
        ax.scatter([rt], [y])
        ax.text(rt, y, name)
    ax.set_xlabel("run_time_s")
    ax.set_ylabel(f"val/fid ({args.reduce})")
    ax.set_title("FID vs wallclock")

    # KID vs wallclock
    ax = axes[1]
    for rt, y, name in zip(wall, kid_y, args.names):
        if y is None:
            continue
        ax.scatter([rt], [y])
        ax.text(rt, y, name)
    ax.set_xlabel("run_time_s")
    ax.set_ylabel(f"val/kid ({args.reduce})")
    ax.set_title("KID vs wallclock")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
