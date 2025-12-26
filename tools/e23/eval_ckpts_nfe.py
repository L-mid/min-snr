#!/usr/bin/env python3
"""
E23 eval-only: load final checkpoints and compute FID/KID + recon@t + sample grid
under a specified sampler + NFE (default: DDIM, NFE=50).

Designed for GPU (Colab/Kaggle). Minimal assumptions about your existing eval pipeline.

Typical usage (baseline):
  python tools/e23/eval_ckpts_nfe.py \
    --label e19_baseline \
    --cfg configs/e19.yaml \
    --ckpt runs/min_snr/...seed=2222/ckpts/last.pt \
    --ckpt runs/min_snr/...seed=3333/ckpts/last.pt \
    --ckpt runs/min_snr/...seed=4444/ckpts/last.pt \
    --out runs/e23/e19_baseline_nfe50 \
    --nfe 50 \
    --fid_stats stats/cifar10_inception_train.npz \
    --fid_n_samples 50000 \
    --kid_n_samples 10000 --kid_subset 1000 --kid_repeats 10 \
    --grid_n 36 \
    --recon_batches 8 --recon_t 10 200 500 700 900 999 \
    --amp

Same again for minsnr checkpoint set (e20/e22/etc).


Testing cli:

python tools/e23/eval_ckpts_nfe.py \
  --label e19_baseline_test_eval \
  --cfg configs/study/MS1_min_snr/e19/e19_best_baseline.yaml \
  --ckpt docs\assets\e19\e19_data\seed_2222_last.pt \
  --ckpt docs\assets\e19\e19_data\seed_3333_last.pt \
  --ckpt docs\assets\e19\e19_data\seed_4444_last.pt \
  --out runs/test_eval_script \
  --nfe 2 \
  --fid_stats stats/cifar10_inception_train.npz \
  --fid_n_samples 2 \
  --kid_n_samples 2 --kid_subset 2 --kid_repeats 2 \
  --kid_cache_real runs/e23/_cache/kid_real_feats_train_seed123.npz \
  --grid_n 2 \
  --recon_batches 2 --recon_t 10 200 500 700 900 999 \
  --use_ema --amp --clip_x0


"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from torchvision import models
from torchvision.utils import save_image

try:
    import yaml
except Exception as e:
    raise RuntimeError("Please `pip install pyyaml`") from e

try:
    from scipy import linalg
except Exception as e:
    raise RuntimeError("Please `pip install scipy` (needed for FID sqrtm).") from e


# ----------------------------
# Model builder (repo-specific)
# ----------------------------

def build_model_from_cfg(cfg: dict) -> torch.nn.Module:
    """
    Tries a couple common paths:
      - ablation_harness.models.build_model(cfg_model_dict)
      - ablation_harness.models.registry.build_model(cfg_model_dict)

    Your cfg should have:
      cfg["model"]["name"] == "unet_cifar32"
      cfg["model"]["base_channels"] (optional)
    """
    model_cfg = cfg.get("model", {})
    # Try common builder entrypoints in your repo
    last_err = None
    for import_path in [
        ("ablation_harness.models", "build_model"),
        ("ablation_harness.models.registry", "build_model"),
    ]:
        mod_name, fn_name = import_path
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name)
            return fn(model_cfg)
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Could not import a model builder. "
        "Expected one of:\n"
        "  from ablation_harness.models import build_model\n"
        "  from ablation_harness.models.registry import build_model\n"
        "Patch build_model_from_cfg() to your repo's constructor.\n"
        f"Last error: {last_err}"
    )


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

def betas_linear(T: int, beta_start=1e-4, beta_end=2e-2) -> np.ndarray:
    return np.linspace(beta_start, beta_end, T, dtype=np.float64)

def betas_cosine(T: int, s: float = 0.008) -> np.ndarray:
    """
    Cosine schedule from Nichol & Dhariwal (Improved DDPM).
    Produces betas via alpha_bar(t) = cos^2((t/T + s)/(1+s) * pi/2)
    """
    steps = np.arange(T + 1, dtype=np.float64)
    t = steps / T
    f = np.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return np.clip(betas, 1e-8, 0.999)

@dataclass
class QSchedule:
    betas: torch.Tensor         # [T]
    alphas: torch.Tensor        # [T]
    alpha_bar: torch.Tensor     # [T]

def make_q(beta_schedule: str, T: int, device: torch.device) -> QSchedule:
    if beta_schedule == "cosine":
        betas = betas_cosine(T)
    elif beta_schedule == "linear":
        betas = betas_linear(T)
    else:
        raise ValueError(f"Unsupported beta_schedule: {beta_schedule}")

    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas, axis=0)

    return QSchedule(
        betas=torch.tensor(betas, device=device, dtype=torch.float32),
        alphas=torch.tensor(alphas, device=device, dtype=torch.float32),
        alpha_bar=torch.tensor(alpha_bar, device=device, dtype=torch.float32),
    )

def make_timesteps(T: int, nfe: int) -> List[int]:
    # integer timesteps, descending, roughly evenly spaced
    ts = np.linspace(0, T - 1, nfe, dtype=np.int64)
    ts = np.unique(ts)  # avoid dupes from rounding
    ts = ts[::-1].tolist()
    if ts[-1] != 0:
        ts.append(0)
    return ts

@torch.inference_mode()
def ddim_sample(
    model: torch.nn.Module,
    q: QSchedule,
    n_samples: int,
    batch_size: int,
    nfe: int,
    device: torch.device,
    amp: bool,
    clip_x0: bool,
) -> torch.Tensor:
    """
    Deterministic DDIM (eta=0). Returns samples in [-1, 1] range.
    """
    T = q.alpha_bar.shape[0]
    timesteps = make_timesteps(T, nfe)

    out = []
    for start in range(0, n_samples, batch_size):
        b = min(batch_size, n_samples - start)
        x = torch.randn(b, 3, 32, 32, device=device)

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1

            a_t = q.alpha_bar[t]                      # scalar
            a_prev = torch.tensor(1.0, device=device) if t_prev < 0 else q.alpha_bar[t_prev]

            t_batch = torch.full((b,), t, device=device, dtype=torch.long)

            with torch.autocast(device_type=device.type, enabled=amp):
                eps = model(x, t_batch)

            sqrt_a_t = torch.sqrt(a_t)
            sqrt_one_minus_a_t = torch.sqrt(1.0 - a_t)
            x0 = (x - sqrt_one_minus_a_t * eps) / (sqrt_a_t + 1e-8)

            if clip_x0:
                x0 = torch.clamp(x0, -1.0, 1.0)

            sqrt_a_prev = torch.sqrt(a_prev)
            sqrt_one_minus_a_prev = torch.sqrt(1.0 - a_prev)

            x = sqrt_a_prev * x0 + sqrt_one_minus_a_prev * eps

        out.append(x.detach().cpu())

    return torch.cat(out, dim=0)


# ----------------------------
# Inception features (for FID/KID)
# ----------------------------

_INCEPTION = None

def get_inception(device: torch.device):
    global _INCEPTION
    if _INCEPTION is None:
        weights = models.Inception_V3_Weights.DEFAULT
        net = models.inception_v3(weights=weights, transform_input=False)
        net.fc = torch.nn.Identity()
        net.eval()
        _INCEPTION = (net, weights)
    net, weights = _INCEPTION
    return net.to(device), weights

def preprocess_for_inception(x01: torch.Tensor, weights) -> torch.Tensor:
    # x01: [B,3,H,W] in [0,1]
    x = F.interpolate(x01, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor(weights.meta["mean"], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(weights.meta["std"], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std

@torch.inference_mode()
def inception_features_from_x(x_minus1_1: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    """
    Compute Inception features for tensor in [-1,1]. Returns [N,2048] float64 numpy.
    """
    net, weights = get_inception(device)
    feats = []

    for start in range(0, x_minus1_1.shape[0], batch_size):
        xb = x_minus1_1[start : start + batch_size].to(device)
        xb01 = torch.clamp((xb + 1.0) / 2.0, 0.0, 1.0)
        xb_in = preprocess_for_inception(xb01, weights)

        out = net(xb_in)
        # Handle torchvision inception output wrappers
        if hasattr(out, "logits"):
            out = out.logits
        elif isinstance(out, (tuple, list)):
            out = out[0]

        feats.append(out.detach().cpu().numpy())

    f = np.concatenate(feats, axis=0).astype(np.float64)
    return f

def fid_from_stats(mu1, sigma1, mu2, sigma2) -> float:
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))

def load_fid_stats(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(str(npz_path))
    for mu_key in ["mu", "mean"]:
        for sig_key in ["sigma", "cov"]:
            if mu_key in d and sig_key in d:
                return d[mu_key].astype(np.float64), d[sig_key].astype(np.float64)
    raise KeyError(f"Unrecognized FID stats keys in {npz_path}. Found: {list(d.keys())}")

def poly_mmd_kid(x: torch.Tensor, y: torch.Tensor, degree=3, gamma=None, coef0=1.0) -> float:
    """
    Unbiased MMD^2 with polynomial kernel.
    x,y: [m,d] float32/float64 torch on same device.
    """
    m = x.shape[0]
    n = y.shape[0]
    d = x.shape[1]
    if gamma is None:
        gamma = 1.0 / d

    Kxx = (gamma * (x @ x.t()) + coef0).pow(degree)
    Kyy = (gamma * (y @ y.t()) + coef0).pow(degree)
    Kxy = (gamma * (x @ y.t()) + coef0).pow(degree)

    # remove diagonals for unbiased estimate
    sum_xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1))
    sum_yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1))
    sum_xy = Kxy.mean()

    mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
    return float(mmd2.item())

def compute_kid(
    f_gen: np.ndarray,
    f_real: np.ndarray,
    subset_size: int,
    repeats: int,
    device: torch.device,
) -> Dict[str, float]:
    rng = np.random.default_rng(12345)
    f_gen_t = torch.tensor(f_gen, device=device, dtype=torch.float32)
    f_real_t = torch.tensor(f_real, device=device, dtype=torch.float32)

    vals = []
    for _ in range(repeats):
        ig = rng.choice(f_gen.shape[0], size=subset_size, replace=False)
        ir = rng.choice(f_real.shape[0], size=subset_size, replace=False)
        vals.append(poly_mmd_kid(f_gen_t[ig], f_real_t[ir]))

    vals = np.array(vals, dtype=np.float64)
    return {"kid_mean": float(vals.mean()), "kid_std": float(vals.std(ddof=1) if len(vals) > 1 else 0.0)}


# ----------------------------
# Recon @ t (predict x0 from eps)
# ----------------------------

def psnr_from_mse(mse: float, max_val: float) -> float:
    return 10.0 * math.log10((max_val * max_val) / max(mse, 1e-12))

@torch.inference_mode()
def recon_profile(
    model: torch.nn.Module,
    q: QSchedule,
    t_values: List[int],
    n_batches: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
    max_val: float,
) -> Dict[str, Dict[str, float]]:
    """
    Uses CIFAR-10 train as a source of x0:
      - x0 in [-1,1]
      - sample x_t
      - predict eps_hat
      - reconstruct x0_hat
      - report MSE/PSNR vs t
    """
    ds = tv.datasets.CIFAR10(root=".", train=True, download=True, transform=tv.transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    stats = {t: {"mse_sum": 0.0, "psnr_sum": 0.0, "n": 0} for t in t_values}

    it = iter(loader)
    for _ in range(n_batches):
        x01, _ = next(it)  # [B,3,32,32] in [0,1]
        x0 = (x01.to(device) * 2.0) - 1.0
        b = x0.shape[0]

        for t in t_values:
            a_t = q.alpha_bar[t]
            eps = torch.randn_like(x0)
            x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1.0 - a_t) * eps

            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            with torch.autocast(device_type=device.type, enabled=amp):
                eps_hat = model(x_t, t_batch)

            x0_hat = (x_t - torch.sqrt(1.0 - a_t) * eps_hat) / (torch.sqrt(a_t) + 1e-8)
            mse = torch.mean((x0_hat - x0) ** 2).item()
            psnr = psnr_from_mse(mse, max_val=max_val)

            stats[t]["mse_sum"] += mse
            stats[t]["psnr_sum"] += psnr
            stats[t]["n"] += 1

    out = {}
    for t in t_values:
        n = stats[t]["n"]
        out[str(t)] = {
            "mse": stats[t]["mse_sum"] / max(n, 1),
            "psnr": stats[t]["psnr_sum"] / max(n, 1),
        }
    return out


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
    # expects each row has fid, kid_mean, kid_std
    out = {}
    for k in ["fid", "kid_mean"]:
        vals = [r[k] for r in rows if k in r]
        if len(vals) > 0:
            out[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
                "n": int(len(vals)),
            }
    return out


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", type=str, required=True)
    ap.add_argument("--cfg", type=str, required=True, help="YAML config (used to build model and schedule).")
    ap.add_argument("--ckpt", type=str, action="append", required=True, help="Path to ckpts/last.pt (repeatable).")
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_ema", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--clip_x0", action="store_true")

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
    ap.add_argument("--kid_cache_real", type=str, default=None, help="Optional .npz to cache real inception features for KID.")

    ap.add_argument("--grid_n", type=int, default=36)

    ap.add_argument("--recon_batches", type=int, default=8)
    ap.add_argument("--recon_batch", type=int, default=64)
    ap.add_argument("--recon_t", type=int, nargs="*", default=[10, 200, 500, 700, 900, 999])
    ap.add_argument("--recon_max_val", type=float, default=2.0)

    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.cfg).read_text())
    beta_schedule = cfg.get("diffusion", {}).get("beta_schedule", "cosine")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # Build once; we reload weights per ckpt
    model = build_model_from_cfg(cfg).to(device)
    model.eval()

    q = make_q(beta_schedule=beta_schedule, T=args.T, device=device)

    # Load real features for KID (cacheable)
    kid_cache_path = Path(args.kid_cache_real) if args.kid_cache_real else None
    if kid_cache_path and kid_cache_path.exists():
        real_feats = np.load(str(kid_cache_path))["feats"].astype(np.float64)
    else:
        # compute real feats from CIFAR-10 train
        ds = tv.datasets.CIFAR10(root=".", train=True, download=True, transform=tv.transforms.ToTensor())
        idx = np.random.default_rng(123).choice(len(ds), size=args.kid_n_samples, replace=False)
        real_imgs = torch.stack([(ds[i][0] * 2.0 - 1.0) for i in idx], dim=0)
        real_feats = inception_features_from_x(real_imgs, device=device, batch_size=args.kid_batch)
        if kid_cache_path:
            kid_cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(kid_cache_path), feats=real_feats)

    fid_mu_real, fid_sig_real = load_fid_stats(Path(args.fid_stats))

    rows = []
    for ckpt_path in args.ckpt:
        ckpt_path = str(ckpt_path)
        seed = parse_seed_from_path(ckpt_path)
        run_tag = f"seed={seed}" if seed is not None else Path(ckpt_path).parent.parent.name

        run_out = out_root / run_tag
        run_out.mkdir(parents=True, exist_ok=True)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)

        if args.use_ema:
            ema = ckpt.get("ema", None)
            if ema is None or "shadow" not in ema:
                raise KeyError(f"--use_ema set but checkpoint has no ema.shadow: {ckpt_path}")
            apply_ema_shadow_(model, ema["shadow"])

        # 1) Sample for FID (and reuse first grid_n for grid)
        x_gen = ddim_sample(
            model=model,
            q=q,
            n_samples=args.fid_n_samples,
            batch_size=args.fid_batch,
            nfe=args.nfe,
            device=device,
            amp=args.amp,
            clip_x0=args.clip_x0,
        )

        # grid
        if args.grid_n > 0:
            grid = x_gen[: args.grid_n]
            grid01 = torch.clamp((grid + 1.0) / 2.0, 0.0, 1.0)
            save_image(grid01, str(run_out / f"grid_nfe{args.nfe}.png"), nrow=int(math.sqrt(args.grid_n)))

        # features for generated
        f_gen = inception_features_from_x(x_gen, device=device, batch_size=args.kid_batch)

        # 2) FID
        mu_gen = np.mean(f_gen, axis=0)
        sig_gen = np.cov(f_gen, rowvar=False)
        fid = fid_from_stats(mu_gen, sig_gen, fid_mu_real, fid_sig_real)

        # 3) KID (use same generated features pool; subset/repeat to reduce variance)
        kid = compute_kid(
            f_gen=f_gen,
            f_real=real_feats,
            subset_size=min(args.kid_subset, f_gen.shape[0], real_feats.shape[0]),
            repeats=args.kid_repeats,
            device=device,
        )

        # 4) Recon profile
        recon = recon_profile(
            model=model,
            q=q,
            t_values=args.recon_t,
            n_batches=args.recon_batches,
            batch_size=args.recon_batch,
            device=device,
            amp=args.amp,
            max_val=args.recon_max_val,
        )

        row = {
            "label": args.label,
            "ckpt": ckpt_path,
            "seed": seed,
            "nfe": args.nfe,
            "beta_schedule": beta_schedule,
            "fid": fid,
            **kid,
            "recon": recon,
        }
        save_json(run_out / "metrics.json", row)
        rows.append(row)

        print(f"[{args.label}] {run_tag}: FID={fid:.4f}  KID={kid['kid_mean']:.6f}±{kid['kid_std']:.6f}")

    summary = {
        "label": args.label,
        "nfe": args.nfe,
        "fid_n_samples": args.fid_n_samples,
        "kid_n_samples": args.kid_n_samples,
        "kid_subset": args.kid_subset,
        "kid_repeats": args.kid_repeats,
        "rows": rows,
        "across_seeds": summarize_across_seeds(rows),
    }
    save_json(out_root / "summary.json", summary)

    print("\n== Across-seeds summary ==")
    print(json.dumps(summary["across_seeds"], indent=2))


if __name__ == "__main__":
    main()
