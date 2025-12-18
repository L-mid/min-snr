#!/usr/bin/env python3
"""
Usage:
  python tools/e12/plot_e12_kid_vs_fid_trajectory.py \
    LOSS_A RESULTS_A \
    LOSS_B RESULTS_B \
    LOSS_C RESULTS_C \
    --names e12a e12b e12c \
    --out docs/assets/e12/e12_plots/e12_kid_vs_fid_trajectory_abc.png


Current:
    python tools/e12/plot_e12_kid_vs_fid_trajectory.py \
    docs/assets/e12/e12a_data/a_loss.jsonl docs/assets/e12/e12a_data/a_results.jsonl \
    docs/assets/e12/e12b_data/b_loss.jsonl docs/assets/e12/e12b_data/b_results.jsonl \
    docs/assets/e12/e12c_data/c_loss.jsonl docs/assets/e12/e12c_data/c_results.jsonl \
    --names e12a-bs4 e12b-bs64 e12c-bs128 \
    --size_by_eval \
    --out docs/assets/e12/e12_plots/e12_kid_vs_fid_trajectory_abc.png


"""

import argparse
import json
from collections import defaultdict
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


def build_step_table(loss_path: str):
    """
    Returns dict step -> dict of keys we care about.
    loss.jsonl often logs val/kid, val/fid, time/eval_s on separate records at same step.
    """
    recs = load_jsonl(loss_path)
    table = defaultdict(dict)
    for r in recs:
        if not isinstance(r, dict):
            continue
        step = r.get("_i", None)
        if step is None:
            continue
        for k in ["val/kid", "val/fid", "time/eval_s"]:
            if k in r:
                table[int(step)][k] = float(r[k])
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="LOSS/RESULTS pairs (LOSS1 RES1 LOSS2 RES2 ...)")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size_by_eval", action="store_true", help="Scale points by time/eval_s at that checkpoint")
    args = ap.parse_args()

    if len(args.paths) % 2 != 0:
        raise SystemExit("Expected even number of positional args: LOSS1 RES1 LOSS2 RES2 ...")
    pairs = [(args.paths[i], args.paths[i + 1]) for i in range(0, len(args.paths), 2)]
    if len(args.names) != len(pairs):
        raise SystemExit(f"--names must have {len(pairs)} entries.")

    plt.figure(figsize=(7, 6))

    for (loss_path, _res_path), name in zip(pairs, args.names):
        tab = build_step_table(loss_path)
        steps = sorted(tab.keys())

        xs, ys, ss = [], [], []
        for s in steps:
            row = tab[s]
            if "val/kid" in row and "val/fid" in row:
                xs.append(row["val/kid"])
                ys.append(row["val/fid"])
                if args.size_by_eval and "time/eval_s" in row:
                    ss.append(max(10.0, 80.0 * row["time/eval_s"]))  # simple scaling
                else:
                    ss.append(40.0)

        if not xs:
            print(f"[warn] no paired (val/kid,val/fid) points for {name}")
            continue

        plt.plot(xs, ys, marker="o", label=name, linewidth=1.5)
        # re-draw points with sizing if requested
        if args.size_by_eval:
            plt.scatter(xs, ys, s=ss, alpha=0.7)

        # mark last point
        plt.scatter([xs[-1]], [ys[-1]], alpha=1.0)

    plt.xlabel("val/kid")
    plt.ylabel("val/fid")
    plt.title("KID vs FID trajectory (checkpoints)")
    plt.legend()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
