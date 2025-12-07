"""
This plot will overlay the loss and fid from loss.jsonl + results.jsonl (if desired)

python tools/plot_loss_fid.py \
  docs/assets/E7/e1_data/loss.jsonl \
  docs/assets/E7/e1_data/results.jsonl \
  docs/assets/E7/e7_data/loss.jsonl \
  docs/assets/E7/e7_data/results.jsonl \
  --out docs/assets/E7/e7_plots/loss_fid_overlay.png

current (with names):
python tools/plot_loss_fid.py \
  docs/assets/e1/e1_data/loss.jsonl docs/assets/e1/e1_data/results.jsonl \
  docs/assets/e2/e2_data/loss.jsonl docs/assets/e2/e2_data/results.jsonl \
  docs/assets/e3/e3_data/loss.jsonl docs/assets/e3/e3_data/results.jsonl \
  docs/assets/e4/e4_data/loss.jsonl docs/assets/e4/e4_data/results.jsonl \
  --names e1-linear-50k e2-minsnr-50k e3-minsnr-10k e4-minsnr-norm-10k \
  --out docs/assets/e4/e4_plots/fid_vs_steps_e1e2e4.png

"""



import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_loss_and_fid(loss_path):
    """Parse loss.jsonl for train loss and any fid-like metrics."""
    steps_loss, losses = [], []
    steps_fid, fids = [], []

    with open(loss_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)

            step = rec.get("_i") or rec.get("global_step") or rec.get("step")
            out = rec.get("out", rec)

            # ---- loss ----
            for key in ("train/loss", "loss", "train_loss"):
                if key in out:
                    if step is not None:
                        steps_loss.append(step)
                        losses.append(out[key])
                    break

            # ---- intermittent FID(s) ----
            for k, v in out.items():
                if "fid" in k.lower():
                    if step is not None:
                        steps_fid.append(step)
                        fids.append(v)
                    break

    return (steps_loss, losses), (steps_fid, fids)


def load_final_fid(results_path, default_step=None):
    """Parse results.jsonl for final FID (take the last FID it sees)."""
    if results_path is None:
        return None, None

    final_step = None
    final_fid = None

    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rec = json.loads(line)
            out = rec.get("out") or rec.get("metrics") or rec

            fid_here = None
            for k, v in out.items():
                if "fid" in k.lower():
                    fid_here = v
                    break

            if fid_here is None:
                continue

            step = rec.get("global_step") or rec.get("step") or rec.get("_i")
            final_step = step
            final_fid = fid_here

    if final_fid is None:
        return None, None

    if final_step is None:
        final_step = default_step
    return final_step, final_fid


def infer_label_from_path(loss_path: str) -> str:
    """Nice default label from the directory name (e.g. 'e1_data')."""
    p = Path(loss_path)
    return p.parent.name or p.stem


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Overlay loss + intermittent FIDs (and final FIDs) "
            "for one or more runs."
        )
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Pairs: loss.jsonl results.jsonl [loss2.jsonl results2.jsonl ...]",
    )
    parser.add_argument(
        "--names",
        nargs="*",
        help="Optional labels per run (same count as pairs).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output PNG path, e.g. docs/assets/E7/e7_plots/loss_fid_overlay.png",
    )
    args = parser.parse_args(argv)

    if len(args.paths) % 2 != 0:
        parser.error(
            "Need an even number of positional paths: "
            "loss.jsonl results.jsonl [loss2.jsonl results2.jsonl ...]"
        )

    num_runs = len(args.paths) // 2
    if args.names and len(args.names) != num_runs:
        parser.error("--names must have the same length as the number of runs")

    fig, ax_loss = plt.subplots(figsize=(7, 4))
    ax_fid = ax_loss.twinx()

    for i in range(num_runs):
        loss_path = args.paths[2 * i]
        results_path = args.paths[2 * i + 1]

        label = (
            args.names[i]
            if args.names and i < len(args.names)
            else infer_label_from_path(loss_path)
        )

        (loss_steps, losses), (fid_steps, fids) = load_loss_and_fid(loss_path)
        default_step = max(loss_steps) if loss_steps else None
        final_step, final_fid = load_final_fid(results_path, default_step)

        # ---- plot loss ----
        if loss_steps:
            ax_loss.plot(
                loss_steps,
                losses,
                linewidth=1.3,
                label=f"loss ({label})",
            )

        # ---- plot intermittent FIDs ----
        if fid_steps:
            ax_fid.plot(
                fid_steps,
                fids,
                "o--",
                markersize=3,
                linewidth=1.0,
                label=f"FID intermittent ({label})",
            )

        # ---- plot final FID ----
        if final_fid is not None:
            ax_fid.scatter(
                [final_step],
                [final_fid],
                marker="*",
                s=100,
                label=f"FID final ({label}) = {final_fid:.2f}",
            )

    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("loss")
    ax_fid.set_ylabel("FID")

    # Merge legends
    lines1, labels1 = ax_loss.get_legend_handles_labels()
    lines2, labels2 = ax_fid.get_legend_handles_labels()
    if lines1 or lines2:
        ax_loss.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="best",
            fontsize=8,
        )

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("Ok.")
    fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()