#!/usr/bin/env python3
"""
Usage:
  python tools/e12/plot_e12_runtime_breakdown.py \
    LOSS_A RESULTS_A \
    LOSS_B RESULTS_B \
    LOSS_C RESULTS_C \
    --names e12a e12b e12c \
    --out docs/assets/e12/e12_plots/e12_runtime_breakdown_abc.png

current:
python tools/e12/plot_e12_runtime_breakdown.py \
  docs/assets/e12/e12a_data/a_loss.jsonl docs/assets/e12/e12a_data/a_results.jsonl \
  docs/assets/e12/e12b_data/b_loss.jsonl docs/assets/e12/e12b_data/b_results.jsonl \
  docs/assets/e12/e12c_data/c_loss.jsonl docs/assets/e12/e12c_data/c_results.jsonl \
  --names e12a-bs4 e12b-bs64 e12c-bs128 \
  --out docs/assets/e12/e12_plots/e12_runtime_breakdown_abc.png    
    
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
    if not recs:
        return {}
    # results.jsonl is typically 1 line; take last dict
    for r in reversed(recs):
        if isinstance(r, dict):
            return r
    return {}


def get(d: dict, key: str, default: float = 0.0) -> float:
    v = d.get(key, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="LOSS/RESULTS pairs (LOSS1 RES1 LOSS2 RES2 ...)")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.paths) % 2 != 0:
        raise SystemExit("Expected even number of positional args: LOSS1 RES1 LOSS2 RES2 ...")

    pairs = [(args.paths[i], args.paths[i + 1]) for i in range(0, len(args.paths), 2)]
    if len(args.names) != len(pairs):
        raise SystemExit(f"--names must have {len(pairs)} entries.")

    labels = args.names

    # Each run: split run_time into train + eval components (+ leftover)
    train_s = []
    modelcopy_s = []
    grid_s = []
    kid_s = []
    fid_s = []
    recon_s = []
    other_s = []
    total_s = []
    eval_frac = []

    for (_loss_path, res_path), name in zip(pairs, labels):
        R = load_single_results(res_path)

        rt = get(R, "run_time_s", default=get(R, "_elapsed_sec", 0.0))

        t_train = get(R, "time/train_s_est", default=0.0)
        t_eval_total = get(R, "time/eval_s_total", default=0.0)

        # Prefer component totals if present; else lump as eval_total
        t_model = get(R, "time/eval/modelcopy_s_total", default=0.0)
        t_grid = get(R, "time/eval/grid_s_total", default=0.0)
        t_kid = get(R, "time/eval/kid_s_total", default=0.0)
        t_fid = get(R, "time/eval/fid_s_total", default=0.0)
        t_recon = get(R, "time/eval/recon_s_total", default=0.0)

        if (t_model + t_grid + t_kid + t_fid + t_recon) <= 0.0 and t_eval_total > 0.0:
            # fallback: treat eval total as a single block (put it in "grid" slot so it still shows)
            t_grid = t_eval_total

        # If train_s_est is missing, infer it from run_time - eval_total (clamped).
        if t_train <= 0.0 and t_eval_total > 0.0:
            t_train = max(0.0, rt - t_eval_total)

        # Any leftover = logging/ckpt/io/etc.
        used = t_train + t_model + t_grid + t_kid + t_fid + t_recon
        t_other = max(0.0, rt - used)

        train_s.append(t_train)
        modelcopy_s.append(t_model)
        grid_s.append(t_grid)
        kid_s.append(t_kid)
        fid_s.append(t_fid)
        recon_s.append(t_recon)
        other_s.append(t_other)
        total_s.append(rt)
        eval_frac.append(get(R, "time/eval_frac", default=(t_eval_total / rt if rt > 0 else 0.0)))

    x = list(range(len(labels)))

    plt.figure(figsize=(11, 5))
    bottom = [0.0] * len(labels)

    def stack(vals, label):
        nonlocal bottom
        plt.bar(x, vals, bottom=bottom, label=label)
        bottom = [b + v for b, v in zip(bottom, vals)]

    stack(train_s, "train")
    stack(modelcopy_s, "eval/modelcopy")
    stack(grid_s, "eval/grid")
    stack(kid_s, "eval/kid")
    stack(fid_s, "eval/fid")
    stack(recon_s, "eval/recon")
    stack(other_s, "other")

    plt.xticks(x, labels, rotation=0)
    plt.ylabel("seconds")
    plt.title("Runtime breakdown (train vs eval vs other)")

    # annotate totals + eval_frac
    for i, (rt, ef) in enumerate(zip(total_s, eval_frac)):
        plt.text(i, rt, f"{rt:.0f}s\n(eval {ef*100:.0f}%)", ha="center", va="bottom")

    plt.legend()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
