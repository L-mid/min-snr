"""
Usage:
python tools/e10/plot_e10_mse_per_t_bins.py \
  docs/assets/e10/e10a_data/loss.jsonl \
  docs/assets/e10/e10b_data/loss.jsonl \
  --names e10a-ddpm e10b-ddim \
  --binsize 50 \
  --out docs/assets/e10/e10_plots/e10_mse_per_t_bins_ab.png
"""
import argparse, json, os, re
import matplotlib.pyplot as plt

RE = re.compile(r"^mse_per_t/mse_t(\d+)$")

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
                r = out
            recs.append(r)
    return recs

def binned_mse(recs, binsize: int):
    # accumulate mse values keyed by bin index
    sums = {}
    cnts = {}
    for r in recs:
        for k, v in r.items():
            m = RE.match(k)
            if not m:
                continue
            t = int(m.group(1))
            b = t // binsize
            sums[b] = sums.get(b, 0.0) + float(v)
            cnts[b] = cnts.get(b, 0) + 1
    xs, ys = [], []
    for b in sorted(cnts.keys()):
        center = b * binsize + 0.5 * (binsize - 1)
        xs.append(center)
        ys.append(sums[b] / max(1, cnts[b]))
    return xs, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("loss_jsonls", nargs="+")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--binsize", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert len(args.loss_jsonls) == len(args.names)

    plt.figure()
    for path, name in zip(args.loss_jsonls, args.names):
        recs = load_jsonl(path)
        x, y = binned_mse(recs, args.binsize)
        plt.plot(x, y, marker="o", label=name)

    plt.xlabel("t (binned)")
    plt.ylabel("mean logged MSE(x0_hat, x0) at sampled t")
    plt.title(f"E10: MSE per t (binsize={args.binsize})")
    plt.legend()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

if __name__ == "__main__":
    main()
