"""
Plotter for hutchison curvator approxmation.


This script generates:

    <out_prefix>_vs_step.png
        X: step (from _i)
        Y₁: curvature/hutch_trace_mean (optionally smoothed)
        Y₂: train/loss on a twin axis

    <out_prefix>_mean_vs_std.png
        X: curvature/hutch_trace_mean
        Y: curvature/hutch_trace_std
        Color: step

    <out_prefix>_vs_grad.png
        X: train/grad_abs_mean (or key set via --grad_key)
        Y: curvature/hutch_trace_mean
        Color: step
        Only uses steps where both metrics exist, so it's robust to sparse logging.


Useage: 
    python tools/hutchinson/plot_curvature.py \
    docs/assets/e5/e5_data/loss.jsonl \
    --name e5-baseline-hutch-10k \
    --out_prefix docs/assets/e5/e5_plots/e5_curvature \
    --smooth_window 21


Current:
    python tools/hutchinson/plot_curvature.py \
    docs/assets/e5/e5_data/loss.jsonl \
    --name e6-baseline-hutch-10k \
    --out_prefix docs/assets/e6/e6_plots/fid_vs_steps_e5e6.png \
    --smooth_window 21


"""


import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np


MetricSeries = Tuple[np.ndarray, np.ndarray]  # (steps, values)


def _read_loss_jsonl(path: Path) -> Dict[str, MetricSeries]:
    """
    Read a loss.jsonl file and return a mapping:
        metric_name -> (steps, values)

    Steps come from top-level "_i" (or "step"/"global_step" fallback).
    Each metric keeps its own step array, so sparsely-logged metrics
    (like curvature) are handled correctly.
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
                # fall back to running index if no explicit step
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
    """Mean generator."""
    if window <= 1 or x.size == 0:
        return x
    window = min(window, x.size)
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="edge")
    cumsum = np.cumsum(padded, dtype=float)
    result = (cumsum[window:] - cumsum[:-window]) / float(window)
    return result


def plot_curvature_vs_step(
    curv_steps: np.ndarray,
    curv_vals: np.ndarray,
    loss_steps: Optional[np.ndarray],
    loss_vals: Optional[np.ndarray],
    name: str,
    out_path: Path,
    smooth_window: int = 1,
):
    """Plots curvature vs steps."""
    curv_s = _rolling_mean(curv_vals, smooth_window)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    ax1.plot(curv_steps[: curv_s.size], curv_s, label=f"{name} curvature", alpha=0.9)
    ax1.set_xlabel("step")
    ax1.set_ylabel("Hutchinson trace (mean)")

    ax2 = ax1.twinx()
    if loss_steps is not None and loss_vals is not None:
        ax2.plot(loss_steps, loss_vals, label=f"{name} loss", alpha=0.5, linestyle="--")
        ax2.set_ylabel("train loss")

    # Combined legend
    lines, labels = [], []
    for ax in (ax1, ax2):
        l, lab = ax.get_legend_handles_labels()
        lines.extend(l)
        labels.extend(lab)
    if lines:
        ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_curvature_mean_vs_std(
    curv_steps: np.ndarray,
    curv_mean: np.ndarray,
    curv_std: np.ndarray,
    name: str,
    out_path: Path,
):
    """Plots curvature mean vs steps."""
    fig, ax = plt.subplots(figsize=(5, 5))

    sc = ax.scatter(curv_mean, curv_std, c=curv_steps, s=10)
    ax.set_xlabel("Hutchinson trace mean")
    ax.set_ylabel("Hutchinson trace std")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("step")

    ax.set_title(name)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_curvature_vs_grad(
    curv_series: MetricSeries,
    grad_series: MetricSeries,
    name: str,
    out_path: Path,
):
    """Plots curvature vs grads. (ensure these are logged on steps together)"""
    curv_steps, curv_vals = curv_series
    grad_steps, grad_vals = grad_series

    # Align on common steps
    curv_map = {int(s): float(v) for s, v in zip(curv_steps, curv_vals)}
    grad_map = {int(s): float(v) for s, v in zip(grad_steps, grad_vals)}
    common_steps = sorted(set(curv_map.keys()) & set(grad_map.keys()))

    if not common_steps:
        print(
            f"[WARN] No common steps between curvature and grad metrics in {name}; "
            "skipping curvature_vs_grad plot."
        )
        return

    xs = np.asarray([grad_map[s] for s in common_steps], dtype=np.float32)
    ys = np.asarray([curv_map[s] for s in common_steps], dtype=np.float32)
    cs = np.asarray(common_steps, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(5, 5))

    sc = ax.scatter(xs, ys, c=cs, s=10)
    ax.set_xlabel("gradient metric")
    ax.set_ylabel("Hutchinson trace mean")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("step")

    ax.set_title(name)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    """Main orchestrator."""
    parser = argparse.ArgumentParser(
        description="Plot curvature diagnostics (Hutchinson trace) from loss.jsonl."
    )
    parser.add_argument(
        "loss_jsonl",
        type=str,
        help="Path to loss.jsonl for a single run.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name for legends/titles (default: inferred from filename).",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        required=True,
        help="Output prefix, e.g. docs/assets/e5/e5_plots/e5_curvature",
    )
    parser.add_argument(
        "--curv_key",
        type=str,
        default="curvature/hutch_trace_mean",
        help="Metric key for Hutchinson trace mean.",
    )
    parser.add_argument(
        "--curv_std_key",
        type=str,
        default="curvature/hutch_trace_std",
        help="Metric key for Hutchinson trace std.",
    )
    parser.add_argument(
        "--loss_key",
        type=str,
        default="train/loss",
        help="Metric key for train loss.",
    )
    parser.add_argument(
        "--grad_key",
        type=str,
        default="train/grad_abs_mean",
        help="Metric key for gradient magnitude (x-axis in third plot).",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=1,
        help="Moving average window for curvature in vs-step plot.",
    )

    args = parser.parse_args()

    loss_path = Path(args.loss_jsonl)
    series = _read_loss_jsonl(loss_path)

    name = args.name or loss_path.stem

    if args.curv_key not in series:
        raise KeyError(f"Missing curvature mean key '{args.curv_key}' in {loss_path}")
    if args.curv_std_key not in series:
        raise KeyError(f"Missing curvature std key '{args.curv_std_key}' in {loss_path}")

    curv_steps, curv_vals = series[args.curv_key]
    curv_std_steps, curv_std_vals = series[args.curv_std_key]

    # Ensure curvature mean/std share the same step grid
    if not np.array_equal(curv_steps, curv_std_steps):
        curv_steps_orig, curv_vals_orig = curv_steps, curv_vals
        curv_std_steps_orig, curv_std_vals_orig = curv_std_steps, curv_std_vals

        step_set = set(curv_steps_orig.tolist()) & set(curv_std_steps_orig.tolist())
        if not step_set:
            raise RuntimeError(
                f"No overlapping steps between {args.curv_key} and {args.curv_std_key}"
            )
        step_list = sorted(step_set)

        def _align(steps_arr: np.ndarray, vals_arr: np.ndarray) -> np.ndarray:
            m = {int(s): float(v) for s, v in zip(steps_arr, vals_arr)}
            return np.asarray([m[s] for s in step_list], dtype=np.float32)

        curv_steps = np.asarray(step_list, dtype=np.int64)
        curv_vals = _align(curv_steps_orig, curv_vals_orig)
        curv_std_vals = _align(curv_std_steps_orig, curv_std_vals_orig)
    else:
        # steps already aligned
        pass

    # Optional series
    loss_series: Optional[MetricSeries] = series.get(args.loss_key)
    grad_series: Optional[MetricSeries] = series.get(args.grad_key)

    out_prefix = Path(args.out_prefix)

    # 1) curvature vs step (+ loss)
    loss_steps = loss_vals = None
    if loss_series is not None:
        loss_steps, loss_vals = loss_series

    plot_curvature_vs_step(
        curv_steps=curv_steps,
        curv_vals=curv_vals,
        loss_steps=loss_steps,
        loss_vals=loss_vals,
        name=name,
        out_path=out_prefix.with_name(out_prefix.name + "_vs_step.png"),
        smooth_window=args.smooth_window,
    )

    # 2) curvature mean vs std
    plot_curvature_mean_vs_std(
        curv_steps=curv_steps,
        curv_mean=curv_vals,
        curv_std=curv_std_vals,
        name=name,
        out_path=out_prefix.with_name(out_prefix.name + "_mean_vs_std.png"),
    )

    # 3) curvature vs grad metric (if available)
    if grad_series is not None:
        plot_curvature_vs_grad(
            curv_series=(curv_steps, curv_vals),
            grad_series=grad_series,
            name=name,
            out_path=out_prefix.with_name(out_prefix.name + "_vs_grad.png"),
        )
    else:
        print(f"[WARN] Gradient key '{args.grad_key}' missing; skipping curvature_vs_grad plot.")

    print(f"Ok, plotting to: {out_prefix.with_name(out_prefix.name)}")

if __name__ == "__main__":
    main()