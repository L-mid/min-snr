
"""
ddim_sample_ckpt.py
Self-contained DDIM sampler that:
- loads a checkpoint (supports your EMA format: {'shadow': list, 'decay', 'step', ...})
- builds betas (linear/cosine) -> precompute_q(betas)
- uses your make_t_schedule(K, nfe)
- runs DDIM sampling with robust float32 math (safe under autocast/AMP elsewhere)
- saves a grid PNG

Usage (example):
  python tools/e17/ddim_sample_ckpt.py \
    --ckpt docs/assets/e17/min-snr-e17-runs/runs/min_snr/min_snr__unet_cifar32__cifar10__adam__lr1e-04__ema1__seed=1/ckpts/last.pt \
    --model-spec "ablation_harness.tasks.diffusion.models.unet_cifar32:UNetCifar32" \
    --model-kw base_channels=64 \
    --schedule cosine --T 1000 --nfe 25 --eta 0.0 \
    --prefer-ema 1 \
    --num-samples 32 --batch-size 64 \
    --out ddim_grid.png

    
Usage (example):
  python tools/e17/ddim_sample_ckpt.py \
    --ckpt docs/assets/e18/runs/min_snr/min_snr__unet_cifar32__cifar10__adam__lr1e-04__ema1__seed=1/ckpts/last.pt \
    --model-spec "ablation_harness.tasks.diffusion.models.unet_cifar32:UNetCifar32" \
    --model-kw base_channels=64 \
    --schedule cosine --T 1000 --nfe 25 --eta 0.0 \
    --prefer-ema 1 \
    --num-samples 32 --batch-size 64 \
    --out ddim_grid.png


"""

import argparse
import importlib
import math
import os
from collections import OrderedDict
from typing import Any, Dict, Tuple, Optional

import torch
from torchvision.utils import save_image, make_grid

from ablation_harness.tasks.diffusion.samplers import DDIMSampler

# ---------------------------
# Your "pieces"
# ---------------------------

@torch.no_grad()
def precompute_q(betas: torch.Tensor):
    alphas = 1.0 - betas  # [K]
    alpha_bar = torch.cumprod(alphas, dim=0)  # [K]

    posterior_variance = torch.zeros_like(betas)  # [K]
    posterior_variance[1:] = betas[1:] * (1 - alpha_bar[:-1]) / (1 - alpha_bar[1:])
    posterior_variance[0] = 1e-20

    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha": torch.sqrt(alphas),
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1 - alpha_bar),
        "posterior_log_var_clipped": torch.log(torch.clamp(posterior_variance, min=1e-20)),
    }


# ---------------------------
# Schedules
# ---------------------------

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    t = steps / T
    alpha_bar = torch.cos(((t + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, min=1e-8, max=0.999)

def build_betas(schedule: str, T: int, beta_start: float, beta_end: float, cosine_s: float) -> torch.Tensor:
    if schedule == "linear":
        return linear_beta_schedule(T, beta_start, beta_end)
    if schedule == "cosine":
        return cosine_beta_schedule(T, cosine_s)
    raise ValueError(schedule)


# ---------------------------
# Checkpoint + EMA handling (your weird EMA)
# ---------------------------

def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if sd and any(k.startswith("module.") for k in sd.keys()):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd

def _extract_model_sd(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    # Common keys in your repo: "model"
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return _strip_module_prefix(ckpt["model"])
    # fallback common names
    for k in ["state_dict", "model_state_dict", "net", "unet"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return _strip_module_prefix(ckpt[k])
    # or ckpt itself might be state_dict
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return _strip_module_prefix(ckpt)
    raise RuntimeError("Could not find model state_dict in checkpoint.")

def _ema_to_state_dict(ema_obj: Any, ref_state_dict: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
    if ema_obj is None:
        return None
    if isinstance(ema_obj, dict) and all(isinstance(v, torch.Tensor) for v in ema_obj.values()):
        return _strip_module_prefix(ema_obj)

    if not (isinstance(ema_obj, dict) and "shadow" in ema_obj):
        return None

    shadow = ema_obj["shadow"]
    if shadow is None:
        return None

    ref_keys = list(ref_state_dict.keys())

    if isinstance(shadow, dict):
        ema_sd = shadow
    elif isinstance(shadow, (list, tuple)):
        if len(shadow) != len(ref_keys):
            raise RuntimeError(f"EMA shadow length {len(shadow)} != model sd length {len(ref_keys)}.")
        ema_sd = OrderedDict((k, v) for k, v in zip(ref_keys, shadow))
    else:
        raise TypeError(f"Unsupported ema['shadow'] type: {type(shadow)}")

    buf_shadow = ema_obj.get("buf_shadow", None)
    if isinstance(buf_shadow, dict):
        ema_sd.update(buf_shadow)

    return _strip_module_prefix(dict(ema_sd))


# ---------------------------
# Model import
# ---------------------------

def parse_kv_list(kvs):
    out: Dict[str, Any] = {}
    for item in kvs or []:
        if "=" not in item:
            raise ValueError(f"Bad --model-kw entry (expected key=value): {item}")
        k, v = item.split("=", 1)
        lv = v.lower()
        if lv in ("true", "false"):
            vv: Any = (lv == "true")
        else:
            try:
                vv = float(v) if "." in v else int(v)
            except ValueError:
                vv = v
        out[k] = vv
    return out

def instantiate_from_spec(spec: str, kwargs: Dict[str, Any]):
    if ":" not in spec:
        raise ValueError("--model-spec must be like 'module.sub:ClassOrFn'")
    mod_name, obj_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, obj_name)
    return obj(**kwargs) if isinstance(obj, type) else obj(**kwargs)


# ---------------------------
# Main
# ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--model-spec", type=str, required=True)
    p.add_argument("--model-kw", action="append", default=[])

    p.add_argument("--schedule", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=2e-2)
    p.add_argument("--cosine-s", type=float, default=0.008)

    p.add_argument("--nfe", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--prefer-ema", type=int, default=1)

    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--channels", type=int, default=3)

    p.add_argument("--no-cuda", action="store_true", help="force CPU even if CUDA is available")
    args = p.parse_args()

    device = torch.device("cpu" if args.no_cuda else args.device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint must be a dict.")

    model_sd = _extract_model_sd(ckpt)
    ema_sd = _ema_to_state_dict(ckpt.get("ema", None), model_sd)

    # Build model
    model_kwargs = parse_kv_list(args.model_kw)
    model = instantiate_from_spec(args.model_spec, model_kwargs).to(device).eval()

    # Choose weights
    use_ema = bool(args.prefer_ema) and (ema_sd is not None)
    sd = ema_sd if use_ema else model_sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("WARNING: load_state_dict(strict=False) mismatches:")
        if missing: print("  missing (first 20):", missing[:20])
        if unexpected: print("  unexpected (first 20):", unexpected[:20])
    print(f"Loaded weights: {'EMA' if use_ema else 'model'}")

    # Build q
    betas = build_betas(args.schedule, args.T, args.beta_start, args.beta_end, args.cosine_s)
    betas = betas.to(device=device, dtype=torch.float32)
    q = precompute_q(betas)

    # Debug: make sure schedule length is T, not nfe
    print("K:", int(q["betas"].numel()), "nfe:", int(args.nfe), "schedule:", args.schedule)
    print("betas[0:3]:", q["betas"][:3].detach().cpu().tolist(), "betas[-3:]:", q["betas"][-3:].detach().cpu().tolist())

    # Sample
    sampler = DDIMSampler(q=q, nfe=args.nfe, eta=args.eta, device=device)

    n = int(args.num_samples)
    Bmax = int(args.batch_size)
    C = int(args.channels)
    H = W = int(args.image_size)

    imgs = []
    remaining = n
    batch_k = 0
    while remaining > 0:
        b = min(Bmax, remaining)
        x = sampler.sample(model, (b, C, H, W), seed=int(args.seed) + batch_k)
        imgs.append(x.detach().cpu())
        remaining -= b
        batch_k += 1

    imgs = torch.cat(imgs, dim=0)  # [-1,1]
    imgs01 = (imgs.clamp(-1, 1) + 1) * 0.5

    # Save grid
    n_show = min(n, 256)
    grid_n = int(math.sqrt(n_show))
    grid_n = max(1, grid_n)
    grid_img = make_grid(imgs01[: grid_n * grid_n], nrow=grid_n, padding=2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_image(grid_img, args.out)
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
