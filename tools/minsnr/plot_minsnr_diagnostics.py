"""
Min-SNR diagnostics plots.

Usage (E3 example):

python tools/minsnr/plot_minsnr_diagnostics.py \
  docs/assets/e4/e4_data/loss.jsonl \
  --out-prefix docs/assets/e4/e4_plots/e4

Outputs:
  e3_early_loss_fid.png
  e3_weight_curve.png
  e3_tmean_hist.png
  e3_tmean_scatter.png
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# I/O + flatten helpers
# ---------------------------------------------------------------------------

def load_flat_loss(path: str) -> List[Dict[str, Any]]:
    """
    Load a loss.jsonl where each line is {"_i": step, "out": {...}} and
    flatten to {"_i": step, **out}.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and "out" in obj:
                flat: Dict[str, Any] = {"_i": obj.get("_i")}
                flat.update(obj["out"])
            else:
                flat = obj
            records.append(flat)
    return records


def infer_step_key(records: Sequence[Dict[str, Any]]) -> Optional[str]:
    """
    Try to guess which key should be used as the x-axis / step.
    For your E3 logs, this resolves to "_i".
    """
    candidates = [
        "global_step",
        "step",
        "train/global_step",
        "train/step",
        "_i",
        "epoch",
    ]
    for key in candidates:
        if any(key in r for r in records):
            return key
    return None


def infer_metric_key(
    records: Sequence[Dict[str, Any]],
    preferred_exact: Sequence[str],
    substring: Optional[str] = None,
) -> Optional[str]:
    """Pick a metric key given an ordered list of preferred names."""
    for name in preferred_exact:
        if any(name in r for r in records):
            return name
    if substring is not None:
        for r in records:
            for k, v in r.items():
                if substring in k and isinstance(v, (int, float)):
                    return k
    return None


def extract_series(
    records: Sequence[Dict[str, Any]],
    step_key: Optional[str],
    metric_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract (steps, values) for a scalar metric.

    Non-scalar entries (lists, dicts) are skipped.
    """
    xs: List[float] = []
    ys: List[float] = []
    for idx, r in enumerate(records):
        if metric_key not in r:
            continue
        y = r[metric_key]
        if isinstance(y, (list, tuple, dict)):
            continue
        if not isinstance(y, (int, float)):
            continue
        if step_key is not None and step_key in r:
            x = r[step_key]
        else:
            x = idx
        xs.append(float(x))
        ys.append(float(y))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_early_loss_fid_plot(
    loss_records: Sequence[Dict[str, Any]],
    out_path: str,
    max_step: int = 10_000,
) -> None:
    """
    Zoomed-in early-phase loss & FID plot.

    - x-axis: training step (from `_i` in your logs).
    - left y-axis: train loss (e.g. 'train/loss').
    - right y-axis: FID at eval steps (e.g. 2k, 4k, … via 'val/fid').

    For E3, FIDs are pulled directly from loss.jsonl.
    """
    if not loss_records:
        print("[early_loss_fid] No loss records, skipping.")
        return

    step_key = infer_step_key(loss_records)
    loss_key = infer_metric_key(
        loss_records,
        preferred_exact=["train/loss", "loss", "train_loss"],
        substring="loss",
    )
    fid_key = infer_metric_key(
        loss_records,
        preferred_exact=["val/fid", "fid"],
        substring="fid",
    )

    if loss_key is None:
        print("[early_loss_fid] Could not find a loss key, skipping.")
        return
    if fid_key is None:
        print("[early_loss_fid] Could not find a FID key in loss.jsonl, skipping.")
        return

    loss_steps, loss_vals = extract_series(loss_records, step_key, loss_key)
    fid_steps, fid_vals = extract_series(loss_records, step_key, fid_key)

    if loss_steps.size == 0 or fid_steps.size == 0:
        print("[early_loss_fid] Empty loss or FID series, skipping.")
        return

    # Restrict loss to early-phase.
    loss_mask = loss_steps <= max_step
    loss_steps_early = loss_steps[loss_mask]
    loss_vals_early = loss_vals[loss_mask]

    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(loss_steps_early, loss_vals_early, label="train loss")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("loss")
    ax1.set_title(f"Early-phase loss & FID (steps ≤ {max_step})")

    ax2 = ax1.twinx()
    ax2.plot(
        fid_steps,
        fid_vals,
        marker="o",
        linestyle="-",
        label="FID",
        alpha=0.8,
    )
    ax2.set_ylabel("FID")

    # Highlight best FID checkpoint with a green star.
    if np.isfinite(fid_vals).any():
        best_idx = int(np.nanargmin(fid_vals))
        best_step = fid_steps[best_idx]
        best_fid = fid_vals[best_idx]

        ax2.scatter(
            [best_step],
            [best_fid],
            marker="*",
            s=180,
            color="green",
            zorder=5,
            label="best ckpt",
        )
        ax2.annotate(
            f"best ckpt @ {int(best_step)}",
            xy=(best_step, best_fid),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Combined legend.
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[early_loss_fid] Wrote {out_path}")


def make_weight_curve_plot(
    loss_records: Sequence[Dict[str, Any]],
    out_path: str,
) -> None:
    """
    Min-SNR weight curve vs t, plus a band showing where t_mean lives.

    For E3:
      - static curve lives in the first record as
          'mins_snr_curve/t', 'mins_snr_curve/weight'
      - per-step stats live as
          'mins_snr/t_mean', 'mins_snr/t_min', 'mins_snr/t_max', ...
    """
    if not loss_records:
        print("[weight_curve] No loss records, skipping.")
        return

    # Find a record with the static curve.
    base: Optional[Dict[str, Any]] = None
    for r in loss_records:
        if "mins_snr_curve/t" in r and "mins_snr_curve/weight" in r:
            base = r
            break

    if base is None:
        print("[weight_curve] Could not find mins_snr_curve/t & weight, skipping.")
        return

    t = np.asarray(base["mins_snr_curve/t"], dtype=float)
    w = np.asarray(base["mins_snr_curve/weight"], dtype=float)

    # Collect t_mean over training.
    t_mean_key = infer_metric_key(
        loss_records,
        preferred_exact=["mins_snr/t_mean"],
        substring="mins_snr/t_mean",
    )
    t_means: Optional[np.ndarray] = None
    if t_mean_key is not None:
        _, t_means = extract_series(loss_records, None, t_mean_key)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(t, w, label="Min-SNR weight(t)")
    ax.set_xlabel("t")
    ax.set_ylabel("weight(t)")
    ax.set_yscale("log")
    ax.set_title("Min-SNR weighting curve")

    if t_means is not None and t_means.size > 0:
        mu = float(np.mean(t_means))
        sigma = float(np.std(t_means))
        band_lo = max(t.min(), mu - sigma)
        band_hi = min(t.max(), mu + sigma)

        ax.axvspan(
            band_lo,
            band_hi,
            alpha=0.2,
            label="t_mean ± 1 std (sampled band)",
        )
        ax.axvline(mu, linestyle="--", alpha=0.5)

        ax.text(
            mu,
            ax.get_ylim()[1],
            f" μ≈{mu:.1f}",
            va="top",
            ha="center",
            fontsize=8,
        )

    ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[weight_curve] Wrote {out_path}")


def make_tmean_plots(
    loss_records: Sequence[Dict[str, Any]],
    out_hist_path: str,
    out_scatter_path: str,
) -> None:
    """
    Histogram of t_mean and scatter of t_mean vs training step.

    This shows “where the optimiser actually spends its time” in timestep space.
    """
    if not loss_records:
        print("[tmean] No loss records, skipping.")
        return

    t_mean_key = infer_metric_key(
        loss_records,
        preferred_exact=["mins_snr/t_mean"],
        substring="mins_snr/t_mean",
    )
    if t_mean_key is None:
        print("[tmean] Could not find mins_snr/t_mean, skipping.")
        return

    step_key = infer_step_key(loss_records)

    steps: List[float] = []
    t_means: List[float] = []

    for idx, r in enumerate(loss_records):
        if t_mean_key not in r:
            continue
        val = r[t_mean_key]
        if not isinstance(val, (int, float)):
            continue
        if step_key is not None and step_key in r:
            x = r[step_key]
        else:
            x = idx
        steps.append(float(x))
        t_means.append(float(val))

    if not t_means:
        print("[tmean] No numeric t_mean values found, skipping.")
        return

    steps_arr = np.asarray(steps, dtype=float)
    t_means_arr = np.asarray(t_means, dtype=float)

    # Histogram
    fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
    ax_hist.hist(t_means_arr, bins=30, alpha=0.8)
    ax_hist.set_xlabel("t_mean")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Histogram of sampled timesteps (t_mean)")
    fig_hist.tight_layout()
    os.makedirs(os.path.dirname(out_hist_path), exist_ok=True)
    fig_hist.savefig(out_hist_path, dpi=300)
    plt.close(fig_hist)
    print(f"[tmean] Wrote {out_hist_path}")

    # Scatter
    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
    ax_scatter.scatter(steps_arr, t_means_arr, s=10, alpha=0.5)
    ax_scatter.set_xlabel("training step")
    ax_scatter.set_ylabel("t_mean")
    ax_scatter.set_title("t_mean over training steps")
    fig_scatter.tight_layout()
    os.makedirs(os.path.dirname(out_scatter_path), exist_ok=True)
    fig_scatter.savefig(out_scatter_path, dpi=300)
    plt.close(fig_scatter)
    print(f"[tmean] Wrote {out_scatter_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Min-SNR diagnostics: early loss/FID, weight curve, timestep stats.",
    )
    parser.add_argument(
        "loss_jsonl",
        type=str,
        help="Path to loss.jsonl for a single run (E3-style logs).",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        required=True,
        help="Prefix for output files (e.g. docs/assets/e3/e3_plots/e3).",
    )
    parser.add_argument(
        "--early-max-step",
        type=int,
        default=10_000,
        help="Max training step for the early-phase loss/FID plot.",
    )

    args = parser.parse_args()

    loss_records = load_flat_loss(args.loss_jsonl)

    early_path = args.out_prefix + "_early_loss_fid.png"
    weights_path = args.out_prefix + "_weight_curve.png"
    tmean_hist_path = args.out_prefix + "_tmean_hist.png"
    tmean_scatter_path = args.out_prefix + "_tmean_scatter.png"

    make_early_loss_fid_plot(
        loss_records=loss_records,
        out_path=early_path,
        max_step=args.early_max_step,
    )
    make_weight_curve_plot(loss_records=loss_records, out_path=weights_path)
    make_tmean_plots(
        loss_records=loss_records,
        out_hist_path=tmean_hist_path,
        out_scatter_path=tmean_scatter_path,
    )


if __name__ == "__main__":
    main()