"""
Plots walltime vs Fid.

Useage:
    python tools/plot_walltime_fid.py \
    docs/assets/e7/e7a_data/loss.jsonl docs/assets/e7/e7a_data/results.jsonl \
    docs/assets/e7/e7b_data/loss.jsonl docs/assets/e7/e7b_data/results.jsonl \
    docs/assets/e7/e7c_data/loss.jsonl docs/assets/e7/e7c_data/results.jsonl \
    --names e7a-baseline-10k e7b-longer-50k e7c-bc64-10k \
    --out docs/assets/e7/e7_plots/fid_vs_walltime_e7abc.png \
    --minutes

"""


import argparse
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

def load_run_time(results_path: Path) -> float:
    """Read total wall time in seconds from results.jsonl."""
    with results_path.open() as f:
        line = f.readline()
    if not line:
        raise RuntimeError(f"{results_path} is empty")

    obj = json.loads(line)
    out = obj.get("out", obj)

    if "run_time_s" in out:
        return float(out["run_time_s"])
    if "_elapsed_sec" in out:
        return float(out["_elapsed_sec"])

    raise KeyError(
        f"No 'run_time_s' or '_elapsed_sec' in {results_path}. "
        "Check your results.jsonl schema."
    )


def load_fid_vs_step(loss_path: Path) -> Tuple[List[int], List[float]]:
    """
    Extract (step, FID) pairs from loss.jsonl.
    Assumes entries with 'val/fid' and global step in '_i'.
    """
    steps = []
    fids = []

    with loss_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out = obj.get("out", {})
            if "val/fid" in out:
                steps.append(int(obj["_i"]))
                fids.append(float(out["val/fid"]))

    if not steps:
        raise RuntimeError(f"No 'val/fid' entries found in {loss_path}")

    return steps, fids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot FID vs (approximate) wall time for multiple runs."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "Sequence of paths: "
            "loss.jsonl results.jsonl [loss.jsonl results.jsonl ...]"
        ),
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Legend names for each run (same length as number of run pairs).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--minutes",
        action="store_true",
        help="Plot time in minutes instead of seconds.",
    )
    args = parser.parse_args()

    if len(args.paths) % 2 != 0:
        raise SystemExit(
            "Error: paths must come in pairs: "
            "loss.jsonl results.jsonl [loss.jsonl results.jsonl ...]"
        )

    num_runs = len(args.paths) // 2
    if len(args.names) != num_runs:
        raise SystemExit(
            f"Error: got {len(args.names)} names but {num_runs} run pairs."
        )

    plt.figure()

    for i in range(num_runs):
        loss_path = Path(args.paths[2 * i])
        results_path = Path(args.paths[2 * i + 1])
        name = args.names[i]

        steps, fids = load_fid_vs_step(loss_path)
        run_time_s = load_run_time(results_path)
        max_step = max(steps)

        # Approximate linear mapping step -> wall time
        times = [run_time_s * (s / max_step) for s in steps]

        if args.minutes:
            times = [t / 60.0 for t in times]
            x_label = "Wall time (minutes)"
        else:
            x_label = "Wall time (seconds)"

        plt.plot(times, fids, marker="o", linewidth=1.5, label=name)

    plt.xlabel(x_label)
    plt.ylabel("FID (lower is better)")
    plt.title("FID vs approximate wall time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()