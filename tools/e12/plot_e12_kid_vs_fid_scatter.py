#!/usr/bin/env python3
"""
Usage:
  python tools/e12/plot_e12_kid_vs_fid_scatter.py \
    LOSS_A RESULTS_A \
    LOSS_B RESULTS_B \
    LOSS_C RESULTS_C \
    --names e12a-bs4-steps10k e12b-bs64-steps625 e12c-bs128-steps313 \
    --out docs/assets/e12/e12_plots/e12_kid_vs_fid_abc.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_jsonl(path: str):
    records = []
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
            records.append(r)
    return records


def pick_key(keys, candidates, substr=None):
    """Pick a key from candidates; else try substring match; else error."""
    for c in candidates:
        if c in keys:
            return c
    if substr:
        hits = sorted([k for k in keys if substr.lower() in k.lower()])
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            # prefer something with "mean"
            mean_hits = [k for k in hits if "mean" in k.lower()]
            if len(mean_hits) == 1:
                return mean_hits[0]
            # else just take the first deterministically
            return hits[0]
    raise KeyError(
        f"Could not find a key. candidates={candidates}, substr={substr}. "
        f"Available keys (sample): {sorted(list(keys))[:50]} ..."
    )


def extract_pairs(loss_path, fid_key, kid_key):
    recs = load_jsonl(loss_path)
    xs, ys = [], []
    for r in recs:
        if not isinstance(r, dict):
            continue
        if fid_key in r and kid_key in r:
            xs.append(r[fid_key])
            ys.append(r[kid_key])
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="LOSS/RESULTS pairs (LOSS1 RES1 LOSS2 RES2 ...)")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--fid_key", default="val/fid")
    ap.add_argument("--kid_key", default=None, help="If omitted, auto-detect val/kid*")

    args = ap.parse_args()

    if len(args.paths) % 2 != 0:
        raise SystemExit("Expected an even number of positional paths: LOSS1 RES1 LOSS2 RES2 ...")

    pairs = [(args.paths[i], args.paths[i + 1]) for i in range(0, len(args.paths), 2)]
    if len(args.names) != len(pairs):
        raise SystemExit(f"--names must have {len(pairs)} entries (got {len(args.names)})")

    # Auto-detect kid key from first loss file if not specified.
    sample_loss = load_jsonl(pairs[0][0])
    sample_keys = set()
    for r in sample_loss:
        if isinstance(r, dict):
            sample_keys |= set(r.keys())

    fid_key = args.fid_key
    if fid_key not in sample_keys:
        # fallback: detect any fid-ish key
        fid_key = pick_key(sample_keys, [args.fid_key], substr="fid")

    kid_key = args.kid_key
    if kid_key is None:
        kid_key = pick_key(sample_keys, [], substr="kid")
    else:
        if kid_key not in sample_keys:
            kid_key = pick_key(sample_keys, [kid_key], substr="kid")

    plt.figure()
    for (loss_path, _res_path), name in zip(pairs, args.names):
        xs, ys = extract_pairs(loss_path, fid_key=fid_key, kid_key=kid_key)
        if not xs:
            print(f"[warn] No paired points found for {name} in {loss_path} using {fid_key} & {kid_key}")
            continue
        plt.scatter(xs, ys, label=name, alpha=0.75)

        # mark last point
        plt.scatter([xs[-1]], [ys[-1]], alpha=1.0)

    plt.xlabel(fid_key)
    plt.ylabel(kid_key)
    plt.title("KID vs FID (checkpoints)")
    plt.legend()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
