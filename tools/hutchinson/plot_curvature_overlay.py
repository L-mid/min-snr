"""
Overlay plotter for Hutchinson curvature vs step.

This script generates a single figure:

    --out <file>.png
        X: step (from _i / step / global_step)
        Y: curvature/hutch_trace_mean (optionally smoothed)
        One line per run (overlay).

Example usage:
    python tools/hutchinson/plot_curvature_overlay.py \
    docs/assets/e5/e5_data/loss.jsonl \
    docs/assets/e6/e6_data/loss.jsonl \
    --names e5-baseline-hutch-10k e6-minsnr-hutch-10k \
    --out docs/assets/e6/e6_plots/curvature_hutch_vs_step_e5e6.png \
    --smooth_window 21

    
Current:
    python tools/hutchinson/plot_curvature_overlay.py \
    docs/assets/e5/e5_data/loss.jsonl \
    docs/assets/e6/e6_data/loss.jsonl \
    --names e5-baseline-hutch-10k e6-minsnr-hutch-10k \
    --out docs/assets/e6/e6_plots/curvature_hutch_vs_step_e5e6.png \
    --smooth_window 21


"""


import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

MetricSeries = Tuple[np.ndarray, np.ndarray]  # (steps, values)


def _read_loss_jsonl(path: Path) -> Dict[str, MetricSeries]:
    """
    Read a loss.jsonl file and return a mapping:
        metric_name -> (steps, values)

    Steps come from top-level "_i" (or "step"/"global_step" fallback).
    Each metric keeps its own step array, so sparsely-logged metrics
    are handled correctly.
    """
    metric_steps: Dict[str, List[int]] = {}
    metric_vals: Dict[str, List[float]] = {}

    with path.open("r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            step = rec.get("_i")
            if step is None:
                step = rec.get("step") or rec.get("global_step")
            if step is None:
                step = line_idx

            step = int(step)
            out = rec.get("out", {})
            for k, v in out.items():
                if isinstance(v, (int, float)):
                    metric_steps.setdefault(k, []).append(step)
                    metric_vals.setdefault(k, []).append(float(v))

    series: Dict[str, MetricSeries] = {}
    for k, vals in metric_vals.items():
        s = np.asarray(metric_steps[k], dtype=np.int64)
        v = np.asarray(vals, dtype=np.float32)
        series[k] = (s, v)

    return series


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average."""
    if window <= 1 or x.size == 0:
        return x
    window = min(window, x.size)
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    cumsum = np.cumsum(padded, dtype=float)
    result = (cumsum[window:] - cumsum[:-window]) / float(window)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Overlay Hutchinson curvature vs step for multiple runs."
    )
    parser.add_argument(
        "loss_jsonl",
        nargs="+",
        help="Paths to loss.jsonl files (one per run).",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Run names (one per loss.jsonl, used in legend).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output PNG path for the overlay figure.",
    )
    parser.add_argument(
        "--curv_key",
        type=str,
        default="curvature/hutch_trace_mean",
        help="Metric key for Hutchinson trace mean.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=1,
        help="Moving average window for curvature in vs-step plot.",
    )

    args = parser.parse_args()

    if len(args.loss_jsonl) != len(args.names):
        raise ValueError(
            f"Got {len(args.loss_jsonl)} loss_jsonl paths but "
            f"{len(args.names)} names; they must match."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))

    for loss_path_str, name in zip(args.loss_jsonl, args.names):
        loss_path = Path(loss_path_str)
        series = _read_loss_jsonl(loss_path)

        if args.curv_key not in series:
            print(f"[WARN] Missing key '{args.curv_key}' in {loss_path}; skipping {name}.")
            continue

        curv_steps, curv_vals = series[args.curv_key]
        curv_s = _rolling_mean(curv_vals, args.smooth_window)
        curv_steps = curv_steps[: curv_s.size]

        ax.plot(curv_steps, curv_s, label=name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Hutchinson trace (mean)")
    ax.set_title("Hutchinson curvature vs step (overlay)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved curvature overlay to: {out_path}")


if __name__ == "__main__":
    main()