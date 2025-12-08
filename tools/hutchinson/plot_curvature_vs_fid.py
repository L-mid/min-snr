"""
Plot curvature vs FID at FID milestones for one or more runs.

For each run, we take:
    - curvature/hutch_trace_mean (from loss.jsonl)
    - val/fid (from the same loss.jsonl)
We align them on step and produce a small curve in FID-vs-curvature space.

This is Plot 4 in the W1-E5/E6 writeup:

    X: Hutchinson trace mean (at FID milestone steps)
    Y: val/fid at those steps
    One polyline per run, with optional step annotations.

Example usage:

    python tools/hutchinson/plot_curvature_vs_fid.py \
        docs/assets/e5/e5_data/loss.jsonl \
        docs/assets/e6/e6_data/loss.jsonl \
        --names e5-baseline-hutch-10k e6-minsnr-hutch-10k \
        --out docs/assets/e6/e6_plots/curvature_vs_fid_e5e6.png \
        --annotate_steps


Current:

    python tools/hutchinson/plot_curvature_vs_fid.py \
        docs/assets/e5/e5_data/loss.jsonl \
        docs/assets/e6/e6_data/loss.jsonl \
        --names e5-baseline-hutch-10k e6-minsnr-hutch-10k \
        --out docs/assets/e6/e6_plots/curvature_vs_fid_e5e6.png \
        --annotate_steps

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


def _extract_curv_and_fid_for_run(
    series: Dict[str, MetricSeries],
    curv_key: str,
    fid_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given metric series from loss.jsonl, extract:
        steps_fid: steps where fid is logged
        curv_at_fid: curvature values at those steps
        fid_vals: fid values
    """
    if curv_key not in series:
        raise KeyError(f"Missing curvature key '{curv_key}' in loss.jsonl")
    if fid_key not in series:
        raise KeyError(f"Missing fid key '{fid_key}' in loss.jsonl")

    curv_steps, curv_vals = series[curv_key]
    fid_steps, fid_vals = series[fid_key]

    # Build a map step -> curvature; curvature is dense (every 100 steps)
    curv_map = {int(s): float(v) for s, v in zip(curv_steps, curv_vals)}

    curv_at_fid: List[float] = []
    fid_used: List[float] = []
    steps_used: List[int] = []

    for s, fid in zip(fid_steps, fid_vals):
        s_int = int(s)
        if s_int not in curv_map:
            # Fallback: nearest step in curvature
            nearest = min(curv_map.keys(), key=lambda t: abs(t - s_int))
            c = curv_map[nearest]
            s_use = nearest
        else:
            c = curv_map[s_int]
            s_use = s_int

        curv_at_fid.append(c)
        fid_used.append(float(fid))
        steps_used.append(s_use)

    curv_arr = np.asarray(curv_at_fid, dtype=np.float32)
    fid_arr = np.asarray(fid_used, dtype=np.float32)
    steps_arr = np.asarray(steps_used, dtype=np.int64)

    # Sort by step just in case
    order = np.argsort(steps_arr)
    return steps_arr[order], curv_arr[order], fid_arr[order]


def main():
    parser = argparse.ArgumentParser(
        description="Plot curvature vs FID for one or more runs from loss.jsonl."
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
        help="Output PNG path for the figure.",
    )
    parser.add_argument(
        "--curv_key",
        type=str,
        default="curvature/hutch_trace_mean",
        help="Metric key for Hutchinson trace mean.",
    )
    parser.add_argument(
        "--fid_key",
        type=str,
        default="val/fid",
        help="Metric key for FID (as logged in loss.jsonl).",
    )
    parser.add_argument(
        "--annotate_steps",
        action="store_true",
        help="If set, annotate each point with its training step.",
    )

    args = parser.parse_args()

    if len(args.loss_jsonl) != len(args.names):
        raise ValueError(
            f"Got {len(args.loss_jsonl)} loss_jsonl paths but "
            f"{len(args.names)} names; they must match."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))

    for loss_path_str, name in zip(args.loss_jsonl, args.names):
        loss_path = Path(loss_path_str)
        series = _read_loss_jsonl(loss_path)

        try:
            steps, curv_vals, fid_vals = _extract_curv_and_fid_for_run(
                series, args.curv_key, args.fid_key
            )
        except KeyError as e:
            print(f"[WARN] {e} in {loss_path}; skipping {name}.")
            continue

        # A small polyline in curvature–FID space
        ax.plot(
            curv_vals,
            fid_vals,
            marker="o",
            linestyle="-",
            label=name,
            alpha=0.9,
        )

        if args.annotate_steps:
            for s, c, f in zip(steps, curv_vals, fid_vals):
                ax.annotate(
                    str(s),
                    xy=(c, f),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=6,
                )

    ax.set_xlabel("Hutchinson trace mean (at FID milestones)")
    ax.set_ylabel("FID")
    ax.set_title("Curvature vs FID at milestones")
    ax.invert_yaxis()  # lower FID is better; visually 'down' is better
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] Saved curvature vs FID plot to: {out_path}")


if __name__ == "__main__":
    main()