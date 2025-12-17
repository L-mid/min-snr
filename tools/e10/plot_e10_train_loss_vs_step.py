"""
Usage:
python tools/e10/plot_e10_train_loss_vs_step.py \
  docs/assets/e10/e10a_data/loss.jsonl \
  docs/assets/e10/e10b_data/loss.jsonl \
  --names e10a-ddpm e10b-ddim \
  --out docs/assets/e10/e10_plots/e10_train_loss_vs_step_ab.png
"""
import argparse, json, os
import matplotlib.pyplot as plt

def load_jsonl(path):
    recs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if isinstance(r, dict) and isinstance(r.get("out"), dict):
                out = r["out"]
                if "step" not in out and "_i" in r:
                    out["step"] = r["_i"]
                r = out
            recs.append(r)
    return recs

def extract_xy(recs, key):
    xs, ys = [], []
    for r in recs:
        if key in r:
            step = r.get("step", r.get("global_step", r.get("_i", None)))
            if step is None:
                continue
            xs.append(int(step))
            ys.append(float(r[key]))
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("loss_jsonls", nargs="+")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert len(args.loss_jsonls) == len(args.names)

    plt.figure()
    for path, name in zip(args.loss_jsonls, args.names):
        recs = load_jsonl(path)
        x, y = extract_xy(recs, "train/loss")
        plt.plot(x, y, label=name)

    plt.xlabel("step")
    plt.ylabel("train/loss")
    plt.title("E10: Train loss vs step")
    plt.legend()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

if __name__ == "__main__":
    main()
