
"""
E23 eval-only: load final checkpoints and compute FID/KID + recon@t + sample grid
under a specified sampler + NFE (default: DDIM, NFE=50).

Designed for GPU (Colab/Kaggle). Minimal assumptions about your existing eval pipeline.

Typical usage (baseline):
    python tools/e23/eval_ckpts_nfe.py \
    --label e19_baseline_eval \
    --cfg configs/study/MS1_min_snr/e19/e19_eval_best_baseline.yaml \
    --ckpt docs/assets/e19/e19_data/seed_2222_last.pt \
    --ckpt docs/assets/e19/e19_data/seed_3333_last.pt \
    --ckpt docs/assets/e19/e19_data/seed_4444_last.pt \
    --out runs/e23/e19_baseline_nfe50 \
    --T 1000 \
    --sampler ddim \
    --nfe 50 \
    --fid_stats stats/cifar10_inception_train.npz \
    --fid_n_samples 50000 --fid_batch 64 \
    --kid_n_samples 10000 --kid_subset 1000 --kid_repeats 10 --kid_batch 128 \
    --grid_n 36 \
    --recon_batches 8 --recon_batch 64 --recon_t 10 200 500 700 900 999 \
    --use_ema

    
    
Same again for minsnr checkpoint set (e20/e22/etc).


Testing cli:

python tools/e23/eval_ckpts_nfe.py \
  --label e19_baseline_test_eval \
  --cfg configs/study/MS1_min_snr/e19/e19_eval_best_baseline.yaml \
  --ckpt docs/assets/e19/e19_data/seed_2222_last.pt \
  --ckpt docs/assets/e19/e19_data/seed_3333_last.pt \
  --ckpt docs/assets/e19/e19_data/seed_4444_last.pt \
  --out runs/test_eval_script \
  --T 1000 \
  --sampler ddim \
  --nfe 2 \
  --fid_stats stats/cifar10_inception_train.npz \
  --fid_n_samples 2 --fid_batch 2 \
  --kid_n_samples 2 --kid_subset 2 --kid_repeats 2 --kid_batch 2 \
  --grid_n 2 \
  --recon_batches 2 --recon_t 10 200 500 700 900 999 \
  --use_ema 

  


"""




from types import SimpleNamespace
import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

try:
    import yaml
except Exception as e:
    raise RuntimeError("Please `pip install pyyaml`") from e

from ablation_harness.tasks.diffusion.models.unet_cifar32 import build_unet_model
from ablation_harness.eval.generative import evaluate_diffusion
from ablation_harness.tasks.diffusion.schedule import precompute_q
from ablation_harness.train import _diffusion_val_recon
import torchvision as tv

# ----------------------------
# Model builder (repo-specific)
# ----------------------------

from types import SimpleNamespace


def build_cifar10_val_loader(*, batch_size: int, num_workers: int):
    # Model expects [-1, 1] if you trained with Normalize(0.5,0.5,0.5)
    tfm = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    ds = tv.datasets.CIFAR10(root="data", train=False, download=True, transform=tfm)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

def make_eval_cfg_from_args(args):
    return SimpleNamespace(
        quick=False,
        sample_seed=0,

        grid=SimpleNamespace(
            enabled=(args.grid_n > 0),
            n_samples=int(args.grid_n),
            batch_size=int(args.fid_batch),   # fine: grid batching
            sampler=str(args.sampler),
            nfe=int(args.nfe),
            save_images=True,
        ),

        kid=SimpleNamespace(
            enabled=(args.kid_n_samples > 0),
            n_samples=int(args.kid_n_samples),
            repeats=int(args.kid_repeats),
            subset_size=int(args.kid_subset),
            sampler=str(args.sampler),
            nfe=int(args.nfe),
            batch_size=int(args.kid_batch),
            real_seed=123,
            real_split="train",
            # if user passes --kid_cache_real, evaluate_diffusion will load it
            feature_cache=(args.kid_cache_real if args.kid_cache_real else None),
        ),

        final=SimpleNamespace(
            enabled=(args.fid_n_samples > 0),
            fid_stats=str(args.fid_stats),
            n_samples=int(args.fid_n_samples),
            sampler=str(args.sampler),
            nfe=int(args.nfe),
            batch_size=int(args.fid_batch),
        ),

        fid_milestone=SimpleNamespace(enabled=False),
    )


def _to_ns(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_to_ns(v) for v in x]
    return x

def build_model_from_cfg(cfg: dict) -> torch.nn.Module:
    cfg_ns = _to_ns(cfg)
    mod = __import__("ablation_harness.tasks.diffusion.models.unet_cifar32", fromlist=["build_unet_model"])
    fn = getattr(mod, "build_unet_model")
    return fn(cfg_ns)


# ----------------------------
# EMA application (matches your ckpt format)
# ----------------------------

def apply_ema_shadow_(model: torch.nn.Module, shadow: List[torch.Tensor]) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) != len(shadow):
        raise ValueError(f"EMA shadow length mismatch: model has {len(params)} params, shadow has {len(shadow)}")
    with torch.no_grad():
        for p, s in zip(params, shadow):
            p.copy_(s.to(device=p.device, dtype=p.dtype))


# ----------------------------
# Diffusion schedule (cosine/linear) and DDIM sampler
# ----------------------------

def make_timesteps(T: int, nfe: int) -> List[int]:
    # integer timesteps, descending, roughly evenly spaced
    ts = np.linspace(0, T - 1, nfe, dtype=np.int64)
    ts = np.unique(ts)  # avoid dupes from rounding
    ts = ts[::-1].tolist()
    if ts[-1] != 0:
        ts.append(0)
    return ts

# ----------------------------
# Helpers
# ----------------------------

def parse_seed_from_path(p: str) -> Optional[int]:
    m = re.search(r"seed[=\-_](\d+)", p)
    return int(m.group(1)) if m else None

def save_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))

def summarize_across_seeds(rows: List[dict]) -> dict:
    out = {}
    for k in ["fid", "kid"]:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals:
            out[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
                "n": int(len(vals)),
            }
    return out


def _strip_module_prefix(state: dict) -> dict:
    # handles DDP checkpoints that save keys like "module.xxx"
    if not state:
        return state
    if any(k.startswith("module.") for k in state.keys()):
        return {k[len("module."):]: v for k, v in state.items()}
    return state

def load_ckpt_into_model_(model: torch.nn.Module, ckpt_path: str, use_ema: bool) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # ---- model weights ----
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        # already looks like a state_dict
        state = ckpt
    else:
        raise RuntimeError(f"Unrecognized checkpoint format in {ckpt_path}. keys={list(ckpt.keys())[:20]}")

    state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=True)

    # ---- EMA (optional) ----
    if use_ema:
        # your apply_ema_shadow_ expects a list of tensors
        shadow = None
        if isinstance(ckpt, dict):
            if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "shadow" in ckpt["ema"]:
                shadow = ckpt["ema"]["shadow"]
            elif "ema_shadow" in ckpt:
                shadow = ckpt["ema_shadow"]

        if shadow is None:
            raise RuntimeError(f"--use_ema was set but EMA shadow not found in {ckpt_path}")

        apply_ema_shadow_(model, shadow)

    return ckpt

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--cfg", type=str, required=True, help="YAML config (used to build model and schedule).")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--ckpt_release_repo", type=str, default=None,
                help="GitHub repo 'owner/repo' that hosts the release assets.")
    ap.add_argument("--ckpt_release_tag", type=str, default=None,
                    help="GitHub release tag (e.g., e19-ckpts).")
    ap.add_argument("--ckpt_cache_dir", type=str, default=None,
                    help="Where to store downloaded ckpts. Default: docs/assets/<label>/<label>_data")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="If set with --ckpt_release_*, infer asset names like seed_{seed}_last.pt.")
    ap.add_argument("--ckpt_pattern", type=str, default=None,
                    help="Optional fnmatch pattern to download multiple assets (e.g. 'seed_*_last.pt').")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--amp", action="store_true")

    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--nfe", type=int, default=50)
    ap.add_argument("--sampler", type=str, default="ddim", choices=["ddim"])  # (this script implements deterministic DDIM)

    ap.add_argument("--fid_stats", type=str, required=True)
    ap.add_argument("--fid_n_samples", type=int, default=50000)
    ap.add_argument("--fid_batch", type=int, default=64)

    ap.add_argument("--kid_n_samples", type=int, default=10000)
    ap.add_argument("--kid_subset", type=int, default=1000)
    ap.add_argument("--kid_repeats", type=int, default=10)
    ap.add_argument("--kid_batch", type=int, default=128)
    ap.add_argument("--kid_cache_real", type=str, default=None,
                help="Optional .npy of cached real inception feats for KID.")
    ap.add_argument("--grid_n", type=int, default=36)

    ap.add_argument("--recon_batches", type=int, default=8)
    ap.add_argument("--recon_batch", type=int, default=64)
    ap.add_argument("--recon_t", type=int, nargs="*", default=[10, 200, 500, 700, 900, 999])
    ap.add_argument("--recon_max_val", type=float, default=2.0)

    args = ap.parse_args()

    out_root = Path(args.out) / args.label
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[e23] out_root={out_root.resolve()}")
    print(f"[e23] n_ckpts={len(args.ckpt)}")

    cfg = yaml.safe_load(Path(args.cfg).read_text())
    from ablation_harness.tasks.diffusion.schedule import get_beta_schedule

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # Build once; we reload weights per ckpt
    model = build_model_from_cfg(cfg).to(device)
    model.eval()
    # --- schedule / q ---
    from ablation_harness.tasks.diffusion.schedule import get_beta_schedule

    beta_schedule = (cfg.get("diffusion", {}) or {}).get("beta_schedule", "cosine")
    betas = get_beta_schedule(beta_schedule, K=int(args.T))

    # ensure torch tensor on correct device (samplers expect torch tensors)
    if not isinstance(betas, torch.Tensor):
        betas = torch.tensor(betas, dtype=torch.float32)
    betas = betas.to(device)

    q = precompute_q(betas=betas)

    rows = []

    # choose a single shared cache dir unless user provided a fixed cache file
    kid_cache_dir = out_root / "_cache" / "kid_real_feats"

    for i, ckpt_path in enumerate(args.ckpt):
        seed = parse_seed_from_path(ckpt_path) or i
        run_out = out_root / f"seed={seed}"
        run_out.mkdir(parents=True, exist_ok=True)

        # load weights (+ EMA if requested)
        load_ckpt_into_model_(model, ckpt_path, use_ema=args.use_ema)
        model.eval()

        # build eval cfg for this run
        eval_cfg = make_eval_cfg_from_args(args)
        eval_cfg.sample_seed = int(seed)

        # only set directory cache if not using an explicit cache file
        if not getattr(eval_cfg.kid, "feature_cache", None):
            eval_cfg.kid.feature_cache_dir = str(kid_cache_dir)

        # run your real pipeline
        res = evaluate_diffusion(
            model_ema=model,
            eval_cfg=eval_cfg,
            q=q,
            out_dir=run_out,
            task=None,
            state_dir=run_out,
            step=None,
        )


        save_json(run_out / "metrics.json", {
            "label": args.label,
            "seed": seed,
            "ckpt": ckpt_path,
            "beta_schedule": beta_schedule,
            "T": int(args.T),
            "sampler": args.sampler,
            "nfe": int(args.nfe),
            **res,
        })

        rows.append({"seed": seed, "ckpt": ckpt_path, **res})


                # ---- recon ----
        if args.recon_batches > 0:
            val_loader = build_cifar10_val_loader(
                batch_size=int(args.recon_batch),
                num_workers=int(cfg.get("data", {}).get("num_workers", 2)),
            )

            recon_cfg = SimpleNamespace(
                log_prefix="val/recon",
                metrics=["mse", "psnr", "l1"],
                max_val=float(args.recon_max_val),
                n_batches=int(args.recon_batches),
                n_images=16,
                save_images=True,
                t_mode="fixed",
                t_values=[int(t) for t in args.recon_t],
            )

            recon_dir = run_out / "recon"
            recon_out = _diffusion_val_recon(
                model_eval=model,
                q=q,
                val_loader=val_loader,
                recon_cfg=recon_cfg,
                device=device,
                out_dir=str(recon_dir),
            )

            # merge into your metrics payload
            res.setdefault("details", {})
            res["details"]["recon"] = recon_out

            save_json(run_out / "recon.json", recon_out)


    # write summary across seeds
    summary = summarize_across_seeds(rows)
    save_json(out_root / "summary.json", {
        "label": args.label,
        "n_ckpts": len(args.ckpt),
        "summary": summary,
        "rows": rows,
    })

if __name__ == "__main__":
    main()
