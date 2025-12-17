"""
Usage:
python tools/e10/plot_e10_recon_profile_vs_t.py \
  docs/assets/e10/e10a_data/loss.jsonl \
  docs/assets/e10/e10b_data/loss.jsonl \
  --names e10a-ddpm e10b-ddim \
  --kind mse \
  --out docs/assets/e10/e10_plots/e10_recon_profile_mse_vs_t_ab.png

# psnr
python tools/e10/plot_e10_recon_profile_vs_t.py \
  docs/assets/e10/e10a_data/loss.jsonl docs/assets/e10/e10b_data/loss.jsonl \
  --names e10a-ddpm e10b-ddim \
  --kind psnr \
  --out docs/assets/e10/e10_plots/e10_recon_profile_psnr_vs_t_ab.png

"""
import argparse, json, os, re
import matplotlib.pyplot as plt

MSE_RE = re.compile(r"^val/recon_mse_t(\d+)$")
PSNR_RE = re.compile(r"^val/recon_psnr_t(\d+)$")

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

def last_recon_record(recs):
    # pick the last dict that contains any recon_t key
    for r in reversed(recs):
        if any(k.startswith("val/recon_mse_t") or k.startswith("val/recon_psnr_t") for k in r.keys()):
            return r
    return None

def extract_profile(r, kind: str):
    rex = MSE_RE if kind == "mse" else PSNR_RE
    pts = []
    for k, v in r.items():
        m = rex.match(k)
        if m:
            t = int(m.group(1))
            pts.append((t, float(v)))
    pts.sort(key=lambda x: x[0])
    return [t for t, _ in pts], [y for _, y in pts]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("loss_jsonls", nargs="+")
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--kind", choices=["mse", "psnr"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert len(args.loss_jsonls) == len(args.names)

    plt.figure()
    for path, name in zip(args.loss_jsonls, args.names):
        recs = load_jsonl(path)
        r = last_recon_record(recs)
        if r is None:
            continue
        ts, ys = extract_profile(r, args.kind)
        step = r.get("step", r.get("_i", None))
        label = f"{name} (step={step})" if step is not None else name
        plt.plot(ts, ys, marker="o", label=label)

    plt.xlabel("t")
    plt.ylabel(f"val/recon_{args.kind}_tXXXX")
    plt.title(f"E10: Recon profile ({args.kind}) vs t (latest logged)")
    plt.legend()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)

if __name__ == "__main__":
    main()
