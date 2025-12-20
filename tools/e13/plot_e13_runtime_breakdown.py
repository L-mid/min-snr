#!/usr/bin/env python3
"""
Plot stacked runtime breakdown bars from one or more results.jsonl files.

Supports either:
  - One results.jsonl containing multiple run records (each line a run), or
  - Multiple results.jsonl files each containing one or more run records.

Example:
python tools/e13/plot_e13_runtime_breakdown.py \
    docs/assets/e13/e13_data/results.jsonl \
    --out docs/assets/e13/e13_plots/e13_runtime_breakdown.png

Or:
  python tools/e13/plot_e13_runtime_breakdown.py \
    docs/assets/e13/e13a_data/a_results.jsonl \
    docs/assets/e13/e13b_data/b_results.jsonl \
    docs/assets/e13/e13c_data/c_results.jsonl \
    --names seed1 seed11 seed1077 \
    --out docs/assets/e13/e13_plots/e13_runtime_breakdown_abc.png
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

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


def get_seed_label(run: Dict[str, Any], fallback: str) -> str:
    cfg = run.get("cfg", {}) or {}
    seed = cfg.get("seed", None)
    if seed is not None:
        return f"seed={seed}"
    # fallback: try to parse digits from fallback
    m = re.search(r"(\d+)", fallback)
    return f"seed={m.group(1)}" if m else fallback


def extract_times(out: Dict[str, Any]) -> Dict[str, float]:
    # Keep these keys aligned with your harness outputs.
    # All values in seconds.
    return {
        "train": float(out.get("time/train_s_est", 0.0) or 0.0),
        "fid": float(out.get("time/eval/fid_s_total", 0.0) or 0.0),
        "kid": float(out.get("time/eval/kid_s_total", 0.0) or 0.0),
        "recon": float(out.get("time/eval/recon_s_total", 0.0) or 0.0),
        "grid": float(out.get("time/eval/grid_s_total", 0.0) or 0.0),
        "modelcopy": float(out.get("time/eval/modelcopy_s_total", 0.0) or 0.0),
    }


def flatten_runs(result_paths: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Returns list of (source_name, run_record).
    Each run record is one JSONL line representing one run.
    """
    runs: List[Tuple[str, Dict[str, Any]]] = []
    for p in result_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        rows = read_jsonl(p)
        # treat each row as a run record
        for i, r in enumerate(rows):
            src = base if len(rows) == 1 else f"{base}#{i}"
            runs.append((src, r))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="One or more results.jsonl paths.")
    ap.add_argument("--names", nargs="*", default=None, help="Optional names for each run (after flattening).")
    ap.add_argument("--out", required=True, help="Output .png path.")
    args = ap.parse_args()

    runs = flatten_runs(args.results)
    if not runs:
        raise SystemExit("No runs found in provided results.jsonl files.")

    # Names: if provided, must match #runs after flattening.
    if args.names is not None and len(args.names) > 0:
        if len(args.names) != len(runs):
            raise SystemExit(f"--names length ({len(args.names)}) must match number of runs ({len(runs)})")
        labels = list(args.names)
    else:
        labels = [get_seed_label(r, src) for (src, r) in runs]

    # Extract times
    comp_order = ["train", "fid", "kid", "recon", "grid", "modelcopy"]
    comp_data = {k: [] for k in comp_order}
    totals = []
    for (_, r) in runs:
        out = r.get("out", {}) or {}
        ts = extract_times(out)
        for k in comp_order:
            comp_data[k].append(ts[k])
        totals.append(sum(ts[k] for k in comp_order))

    x = np.arange(len(labels))
    bottom = np.zeros(len(labels), dtype=np.float64)

    plt.figure(figsize=(10, 5))
    for k in comp_order:
        vals = np.array(comp_data[k], dtype=np.float64)
        plt.bar(x, vals, bottom=bottom, label=k)
        bottom += vals

    plt.xticks(x, labels, rotation=0)
    plt.ylabel("seconds")
    plt.title("Runtime breakdown (stacked)")
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    # Print a quick text summary too (handy for terminal logs).
    for lab, tot in zip(labels, totals):
        print(f"{lab}: total_plotted_s={tot:.2f}")


if __name__ == "__main__":
    main()
