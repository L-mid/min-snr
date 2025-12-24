#!/usr/bin/env python3
"""
Plots from one or more loss.jsonl files:
  - FID vs step (line plot)
  - KID vs step (line plot)
  - Eval cost vs step (avg over runs): fid_s, kid_s, eval_total_s
  - KID vs FID scatter (sanity / correlation check)

Example:
  python tools/e13/plot_e13_metrics_and_costs.py \
    docs/assets/e13/e13a_data/a_loss.jsonl \
    docs/assets/e13/e13b_data/b_loss.jsonl \
    docs/assets/e13/e13c_data/c_loss.jsonl \
    --names seed1 seed11 seed1077 \
    --out_dir docs/assets/e13/e13_plots/


current:

  python tools/e13/plot_e13_metrics_and_costs.py \
    docs/assets/e13/e13_data/1_loss.jsonl \
    docs/assets/e13/e13_data/11_loss.jsonl \
    docs/assets/e13/e13_data/1077_loss.jsonl \
    --names seed1 seed11 seed1077 \
    --out_dir docs/assets/e13/e13_plots/
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def infer_run_id(rows: List[Dict[str, Any]], path: str) -> str:
    # Prefer cfg.seed if present anywhere.
    for r in rows:
        cfg = r.get("cfg", None)
        if isinstance(cfg, dict) and "seed" in cfg:
            return f"seed={cfg['seed']}"
    # fallback: digits from filename
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)", stem)
    return f"seed={m.group(1)}" if m else stem


@dataclass
class Milestone:
    step: int
    fid: Optional[float]
    kid: Optional[float]
    fid_s: float
    kid_s: float
    eval_s: float


def extract_milestones(loss_path: str) -> Tuple[str, List[Milestone]]:
    rows = read_jsonl(loss_path)
    run_id = infer_run_id(rows, loss_path)

    ms: List[Milestone] = []
    for r in rows:
        step = r.get("_i", None)
        out = r.get("out", None)

        if not isinstance(step, int) or not isinstance(out, dict):
            continue

        fid = out.get("val/fid", None)
        kid = out.get("val/kid", None)

        # Only keep eval milestones (where at least one metric exists).
        if fid is None and kid is None:
            continue

        ms.append(
            Milestone(
                step=step,
                fid=float(fid) if fid is not None else None,
                kid=float(kid) if kid is not None else None,
                fid_s=float(out.get("time/eval/fid_s", 0.0) or 0.0),
                kid_s=float(out.get("time/eval/kid_s", 0.0) or 0.0),
                eval_s=float(out.get("time/eval_s", 0.0) or 0.0),
            )
        )

    ms.sort(key=lambda m: m.step)
    return run_id, ms


def to_arrays(ms: List[Milestone]) -> Dict[str, np.ndarray]:
    step = np.array([m.step for m in ms], dtype=np.int64)
    fid = np.array([np.nan if m.fid is None else m.fid for m in ms], dtype=np.float64)
    kid = np.array([np.nan if m.kid is None else m.kid for m in ms], dtype=np.float64)
    fid_s = np.array([m.fid_s for m in ms], dtype=np.float64)
    kid_s = np.array([m.kid_s for m in ms], dtype=np.float64)
    eval_s = np.array([m.eval_s for m in ms], dtype=np.float64)
    return {"step": step, "fid": fid, "kid": kid, "fid_s": fid_s, "kid_s": kid_s, "eval_s": eval_s}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("loss", nargs="+", help="One or more loss.jsonl paths.")
    ap.add_argument("--names", nargs="*", default=None, help="Optional run names (must match #loss files).")
    ap.add_argument("--out_dir", required=True, help="Directory to write .png outputs.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.names is not None and len(args.names) > 0 and len(args.names) != len(args.loss):
        raise SystemExit(f"--names length ({len(args.names)}) must match number of loss files ({len(args.loss)})")

    runs: List[Tuple[str, Dict[str, np.ndarray]]] = []
    for i, p in enumerate(args.loss):
        run_id, ms = extract_milestones(p)
        if args.names is not None and len(args.names) > 0:
            run_id = args.names[i]
        arr = to_arrays(ms)
        runs.append((run_id, arr))

    if not runs:
        raise SystemExit("No milestones found (val/fid or val/kid) in provided loss.jsonl files.")

    # -------- Plot 1: FID vs step --------
    out_fid = os.path.join(args.out_dir, "e13_fid_vs_step.png")
    plt.figure(figsize=(7.5, 4.8))
    for run_id, arr in runs:
        if np.all(np.isnan(arr["fid"])):
            continue
        plt.plot(arr["step"], arr["fid"], marker="o", label=run_id)
    plt.xlabel("step")
    plt.ylabel("val/fid")
    plt.title("FID vs step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_fid, dpi=200)
    plt.close()

    # -------- Plot 2: KID vs step --------
    out_kid = os.path.join(args.out_dir, "e13_kid_vs_step.png")
    plt.figure(figsize=(7.5, 4.8))
    for run_id, arr in runs:
        if np.all(np.isnan(arr["kid"])):
            continue
        plt.plot(arr["step"], arr["kid"], marker="o", label=run_id)
    plt.xlabel("step")
    plt.ylabel("val/kid")
    plt.title("KID vs step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_kid, dpi=200)
    plt.close()

    # -------- Plot 3: Eval cost vs step (avg over runs) --------
    # Align by step; average fid_s/kid_s/eval_s where steps overlap.
    # Build a map step -> list of values.
    step_to_vals: Dict[int, Dict[str, List[float]]] = {}
    for _, arr in runs:
        for s, fid_s, kid_s, eval_s in zip(arr["step"], arr["fid_s"], arr["kid_s"], arr["eval_s"]):
            s_int = int(s)
            if s_int not in step_to_vals:
                step_to_vals[s_int] = {"fid_s": [], "kid_s": [], "eval_s": []}
            step_to_vals[s_int]["fid_s"].append(float(fid_s))
            step_to_vals[s_int]["kid_s"].append(float(kid_s))
            step_to_vals[s_int]["eval_s"].append(float(eval_s))

    steps_sorted = np.array(sorted(step_to_vals.keys()), dtype=np.int64)
    fid_s_mean = np.array([np.mean(step_to_vals[int(s)]["fid_s"]) for s in steps_sorted], dtype=np.float64)
    kid_s_mean = np.array([np.mean(step_to_vals[int(s)]["kid_s"]) for s in steps_sorted], dtype=np.float64)
    eval_s_mean = np.array([np.mean(step_to_vals[int(s)]["eval_s"]) for s in steps_sorted], dtype=np.float64)

    out_cost = os.path.join(args.out_dir, "e13_eval_cost_vs_step_avg.png")
    plt.figure(figsize=(7.5, 4.8))
    plt.plot(steps_sorted, fid_s_mean, marker="o", label="fid_s")
    plt.plot(steps_sorted, kid_s_mean, marker="o", label="kid_s")
    plt.plot(steps_sorted, eval_s_mean, marker="o", label="eval_total_s")
    plt.xlabel("step")
    plt.ylabel("seconds")
    plt.title("Eval cost vs step (avg over runs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_cost, dpi=200)
    plt.close()

    # -------- Plot 4: KID vs FID scatter --------
    out_scatter = os.path.join(args.out_dir, "e13_kid_vs_fid_scatter.png")
    plt.figure(figsize=(6.8, 5.2))
    for run_id, arr in runs:
        # only plot points where both exist
        mask = ~np.isnan(arr["fid"]) & ~np.isnan(arr["kid"])
        if not np.any(mask):
            continue
        plt.scatter(arr["fid"][mask], arr["kid"][mask], label=run_id)
    plt.xlabel("val/fid (lower is better)")
    plt.ylabel("val/kid (lower is better)")
    plt.title("KID vs FID scatter")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_scatter, dpi=200)
    plt.close()

    print("Wrote:")
    print(" ", out_fid)
    print(" ", out_kid)
    print(" ", out_cost)
    print(" ", out_scatter)


if __name__ == "__main__":
    main()
