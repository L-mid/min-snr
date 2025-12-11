import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

def load_jsonl(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def extract_step(rec):
    return rec.get("_i") or rec.get("step") or rec.get("global_step")


def collect_loss_series(loss_path, results_path):
    loss_recs = load_jsonl(loss_path)
    res_recs = load_jsonl(results_path)

    # Loss & grad & curvature vs step
    steps_loss = []
    losses = []
    grads = []
    curv = []

    for r in loss_recs:
        step = extract_step(r)
        if step is None:
            continue
        if "train/loss" in r:
            steps_loss.append(step)
            losses.append(r["train/loss"])
            grads.append(r.get("train/grad_global_L2"))
            curv.append(r.get("curvature/hutch_trace_mean"))

    # FID vs step
    steps_fid = []
    fids = []
    for r in res_recs:
        step = extract_step(r)
        if step is None:
            continue
        if "val/fid" in r:
            steps_fid.append(step)
            fids.append(r["val/fid"])

    return {
        "steps_loss": steps_loss,
        "loss": losses,
        "grad": grads,
        "curv": curv,
        "steps_fid": steps_fid,
        "fid": fids,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss/FID/grad/curvature vs step (γ overlay)."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Pairs of loss.jsonl results.jsonl for each experiment.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        required=True,
        help="Names/labels for each experiment (same order as file pairs).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output PNG path.",
    )
    args = parser.parse_args()

    if len(args.files) % 2 != 0:
        raise ValueError("Provide loss/results in pairs: LOSS_A RESULTS_A LOSS_B RESULTS_B ...")

    n_exp = len(args.files) // 2
    if len(args.names) != n_exp:
        raise ValueError("Need one --names entry per loss/results pair.")

    series_list = []
    for i in range(n_exp):
        loss_path = args.files[2 * i]
        results_path = args.files[2 * i + 1]
        series_list.append(collect_loss_series(loss_path, results_path))

    fig, axes = plt.subplots(4, 1, figsize=(8, 12), sharex=True)
    ax_loss, ax_fid, ax_grad, ax_curv = axes

    for name, s in zip(args.names, series_list):
        # Loss
        ax_loss.plot(s["steps_loss"], s["loss"], label=name)
        # FID
        if s["steps_fid"]:
            ax_fid.plot(s["steps_fid"], s["fid"], marker="o", linestyle="-", label=name)
        # Grad
        if any(g is not None for g in s["grad"]):
            grad_steps = [step for step, g in zip(s["steps_loss"], s["grad"]) if g is not None]
            grad_vals = [g for g in s["grad"] if g is not None]
            ax_grad.plot(grad_steps, grad_vals, label=name)
        # Curvature
        if any(c is not None for c in s["curv"]):
            curv_steps = [step for step, c in zip(s["steps_loss"], s["curv"]) if c is not None]
            curv_vals = [c for c in s["curv"] if c is not None]
            ax_curv.plot(curv_steps, curv_vals, label=name)

    ax_loss.set_ylabel("train/loss")
    ax_loss.legend()
    ax_loss.grid(True)

    ax_fid.set_ylabel("val/FID")
    ax_fid.legend()
    ax_fid.grid(True)

    ax_grad.set_ylabel("grad_global_L2")
    ax_grad.legend()
    ax_grad.grid(True)

    ax_curv.set_ylabel("Hutch trace")
    ax_curv.set_xlabel("global step")
    ax_curv.legend()
    ax_curv.grid(True)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
