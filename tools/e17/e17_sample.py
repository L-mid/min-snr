#!/usr/bin/env python3
"""
ddim_sample.py — single-file DDIM sampler with *linear* or *cosine* beta schedules
and a controllable NFE (number of sampling steps).

Assumptions:
- Your model predicts epsilon: eps = model(x_t, t) where:
  - x_t: (B,C,H,W) float tensor in [-1,1] space
  - t: (B,) int64 tensor with values in [0, T-1]
- Your checkpoint contains a model state_dict (or has it under a common key).

Example:
  python tools/e17/e17_sample.py \
    --ckpt docs/assets/e17/min-snr-e17-runs/runs/min_snr/min_snr__unet_cifar32__cifar10__adam__lr1e-04__ema1__seed=1/ckpts/last.pt \
    --model-spec "ablation_harness.tasks.diffusion.models.unet_cifar32:UNetCifar32" \
    --model-kw base_channels=64 \
    --schedule linear \
    --nfe 25 \
    --eta 0.0 \
    --num-samples 32 \
    --batch-size 64 \
    --out samples_cosine_but_linear_no_ema_nfe25.png \
    --prefer-ema 0


    
Current:
  python tools/e17/e17_sample.py \
    --ckpt docs/assets/e16/runs/min_snr/min_snr__unet_cifar32__cifar10__adam__lr1e-04__ema1__seed=1/ckpts/last.pt \
    --model-spec "ablation_harness.tasks.diffusion.models.unet_cifar32:UNetCifar32" \
    --model-kw base_channels=64 \
    --schedule linear \
    --nfe 25 \
    --eta 0.0 \
    --num-samples 32 \
    --batch-size 64 \
    --out samples_linear_nfe25.png \
    --prefer-ema 1



If your checkpoint has EMA weights, add:
  --prefer-ema 1
"""

import argparse
import importlib
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from collections import OrderedDict
import torch
from torchvision.utils import save_image, make_grid


# ---------------------------
# Schedules
# ---------------------------

def linear_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    # Classic DDPM linear schedule
    return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    # Nichol & Dhariwal (Improved DDPM) cosine schedule
    # alpha_bar(t) = cos^2(((t/T)+s)/(1+s) * pi/2)
    steps = torch.arange(T + 1, dtype=torch.float64)
    t = steps / T
    alpha_bar = torch.cos(((t + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]

    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    # Clamp for numerical stability
    return torch.clamp(betas, min=1e-8, max=0.999)

def make_diffusion_buffers(T: int, schedule: str, device: torch.device) -> Dict[str, torch.Tensor]:
    if schedule == "linear":
        betas = linear_beta_schedule(T)
    elif schedule == "cosine":
        betas = cosine_beta_schedule(T)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    betas = betas.to(torch.float64)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)

    # Common precomputes (float64 for accuracy; cast later per device)
    out = {
        "betas": betas,
        "alphas": alphas,
        "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_one_minus_alpha_bar": torch.sqrt(1.0 - alpha_bar),
    }
    # Move to device + float32 for model compute
    return {k: v.to(device=device, dtype=torch.float32) for k, v in out.items()}


# ---------------------------
# DDIM Sampler
# ---------------------------

@torch.no_grad()
def ddim_sample(
    model,
    *,
    shape: Tuple[int, int, int, int],
    T: int,
    nfe: int,
    schedule: str,
    eta: float,
    device: torch.device,
    clamp_x0: bool = True,
    autocast: bool = True,
) -> torch.Tensor:
    """
    Deterministic DDIM when eta=0. Stochastic DDIM when eta>0.
    Uses a uniform stride of timesteps from T-1 down to 0 with length nfe.
    """
    assert nfe >= 2, "nfe must be >= 2"

    buf = make_diffusion_buffers(T=T, schedule=schedule, device=device)
    alpha_bar = buf["alpha_bar"]  # (T,)
    sqrt_alpha_bar = buf["sqrt_alpha_bar"]
    sqrt_1m_alpha_bar = buf["sqrt_one_minus_alpha_bar"]

    # Timesteps to visit (descending)
    # e.g., for T=1000,nfe=50 => 50 steps from 999..0
    t_seq = torch.linspace(T - 1, 0, steps=nfe, device=device).long()
    # Previous timesteps (t_{i+1} in the sequence). For last step, treat t_prev = -1 => alpha_bar_prev = 1
    t_prev_seq = torch.cat([t_seq[1:], torch.tensor([-1], device=device, dtype=torch.long)], dim=0)

    x = torch.randn(shape, device=device, dtype=torch.float32)

    use_amp = (autocast and device.type == "cuda")
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else torch.autocast(device_type="cpu", enabled=False)

    for i in range(nfe):
        t = t_seq[i]
        t_prev = t_prev_seq[i]

        a_bar_t = alpha_bar[t]              # scalar
        sqrt_a_bar_t = sqrt_alpha_bar[t]
        sqrt_1m_a_bar_t = sqrt_1m_alpha_bar[t]

        if t_prev.item() >= 0:
            a_bar_prev = alpha_bar[t_prev]
        else:
            a_bar_prev = torch.tensor(1.0, device=device, dtype=torch.float32)

        # Model predicts eps(x_t, t)
        t_batch = torch.full((shape[0],), int(t.item()), device=device, dtype=torch.long)

        with amp_ctx:
            eps = model(x, t_batch)

        # x0 prediction: x0 = (x_t - sqrt(1-a_bar_t)*eps) / sqrt(a_bar_t)
        x0 = (x - sqrt_1m_a_bar_t * eps) / torch.clamp(sqrt_a_bar_t, min=1e-12)

        if clamp_x0:
            x0 = torch.clamp(x0, -1.0, 1.0)

        # DDIM sigma (from DDIM paper)
        # sigma_t = eta * sqrt((1-a_bar_prev)/(1-a_bar_t)) * sqrt(1 - a_bar_t/a_bar_prev)
        # When eta=0 => deterministic
        one = torch.tensor(1.0, device=device, dtype=torch.float32)
        denom = torch.clamp(one - a_bar_t, min=1e-12)
        frac = torch.clamp((one - a_bar_prev) / denom, min=0.0)
        inner = torch.clamp(one - (a_bar_t / torch.clamp(a_bar_prev, min=1e-12)), min=0.0)
        sigma_t = float(eta) * torch.sqrt(frac) * torch.sqrt(inner)

        # Direction term coefficient:
        # sqrt(1 - a_bar_prev - sigma_t^2)
        c = torch.sqrt(torch.clamp(one - a_bar_prev - sigma_t * sigma_t, min=0.0))

        if float(eta) > 0.0:
            z = torch.randn_like(x)
        else:
            z = torch.zeros_like(x)

        # x_{t-1} = sqrt(a_bar_prev)*x0 + c*eps + sigma*z
        x = torch.sqrt(torch.clamp(a_bar_prev, min=0.0)) * x0 + c * eps + sigma_t * z

    return x


# ---------------------------
# Checkpoint loading helpers
# ---------------------------


def _strip_module_prefix(sd):
    if sd and any(k.startswith("module.") for k in sd.keys()):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def _ema_to_state_dict(ema_obj, ref_state_dict):
    """
    Handles EMA formats like:
      ema = {'decay':..., 'step':..., 'shadow': [t0,t1,...], 'buf_shadow': ...}
    where shadow is stored as a list aligned with ref_state_dict.keys().
    """
    if ema_obj is None:
        return None

    # If it's already a tensor-keyed dict, treat as state_dict.
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
            raise RuntimeError(
                f"EMA shadow length {len(shadow)} != ref_state_dict length {len(ref_keys)}. "
                f"Can't align EMA weights."
            )
        ema_sd = OrderedDict((k, v) for k, v in zip(ref_keys, shadow))
    else:
        raise TypeError(f"Unsupported ema['shadow'] type: {type(shadow)}")

    # Optional buffer shadow (yours is None, but handle it anyway)
    buf_shadow = ema_obj.get("buf_shadow", None)
    if isinstance(buf_shadow, dict):
        ema_sd.update(buf_shadow)

    return _strip_module_prefix(ema_sd)


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Checkpoint must be a dict.")

    # Your checkpoint uses 'model' and 'ema'
    model_sd = ckpt.get("model", None)
    if model_sd is None:
        raise RuntimeError("No 'model' key found in checkpoint.")

    model_sd = _strip_module_prefix(model_sd)

    ema_sd = _ema_to_state_dict(ckpt.get("ema", None), model_sd)
    return model_sd, ema_sd, ckpt


# ---------------------------
# Dynamic model import
# ---------------------------

def parse_kv_list(kvs):
    out: Dict[str, Any] = {}
    for item in kvs or []:
        if "=" not in item:
            raise ValueError(f"Bad --model-kw entry (expected key=value): {item}")
        k, v = item.split("=", 1)
        # basic type parsing
        if v.lower() in ("true", "false"):
            vv: Any = (v.lower() == "true")
        else:
            try:
                if "." in v:
                    vv = float(v)
                else:
                    vv = int(v)
            except ValueError:
                vv = v
        out[k] = vv
    return out

def instantiate_from_spec(spec: str, kwargs: Dict[str, Any]):
    """
    spec like "my_pkg.my_mod:MyModel" or "my_pkg.my_mod:build_model"
    """
    if ":" not in spec:
        raise ValueError("--model-spec must be like 'module.submodule:CallableOrClassName'")
    mod_name, obj_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, obj_name)

    if isinstance(obj, type):
        return obj(**kwargs)
    if callable(obj):
        return obj(**kwargs)
    raise ValueError(f"Object {obj_name} from {mod_name} is not callable/class.")


# ---------------------------
# Main
# ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt/.pth)")
    p.add_argument("--out", type=str, required=True, help="Output image file (png/jpg)")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)

    # model
    p.add_argument("--model-spec", type=str, required=True,
                   help="Python spec 'module.submodule:ClassOrBuilder' to construct the model")
    p.add_argument("--model-kw", action="append", default=[],
                   help="Repeatable key=value kwargs passed to the model constructor (e.g. base_channels=64)")

    # diffusion + sampling
    p.add_argument("--schedule", type=str, choices=["linear", "cosine"], default="cosine")
    p.add_argument("--T", type=int, default=1000, help="Training diffusion steps (schedule length)")
    p.add_argument("--nfe", type=int, default=50, help="Number of DDIM sampling steps")
    p.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0=deter, >0 stochastic)")
    p.add_argument("--clamp-x0", type=int, default=1, help="Clamp x0 to [-1,1] during sampling (1/0)")
    p.add_argument("--no-amp", action="store_true", help="Disable autocast AMP on CUDA")

    # sampling batch
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--channels", type=int, default=3)

    # checkpoint preference
    p.add_argument("--prefer-ema", type=int, default=0, help="If checkpoint has EMA weights, use them (1/0)")

    args = p.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model_kwargs = parse_kv_list(args.model_kw)

    # Build model
    model = instantiate_from_spec(args.model_spec, model_kwargs)
    model.to(device)
    model.eval()

    # Load weights
    model_sd, ema_sd, raw = load_checkpoint(args.ckpt)
    if args.prefer_ema and (ema_sd is not None):
        sd_to_load = ema_sd
        which = "EMA"
    else:
        sd_to_load = model_sd
        which = "model"

    missing, unexpected = model.load_state_dict(sd_to_load, strict=False)
    if missing or unexpected:
        print("WARNING: load_state_dict(strict=False) had mismatches:")
        if missing:
            print("  missing keys (first 20):", missing[:20])
        if unexpected:
            print("  unexpected keys (first 20):", unexpected[:20])
        print("If this is not expected, fix --model-spec/--model-kw to match the checkpoint.")

    print(f"Loaded {which} weights from: {args.ckpt}")

    # Sample in batches
    n = args.num_samples
    bs = min(args.batch_size, n)
    H = W = args.image_size
    C = args.channels

    all_imgs = []
    steps = (n + bs - 1) // bs

    for i in range(steps):
        cur = min(bs, n - i * bs)
        x = ddim_sample(
            model,
            shape=(cur, C, H, W),
            T=args.T,
            nfe=args.nfe,
            schedule=args.schedule,
            eta=args.eta,
            device=device,
            clamp_x0=bool(args.clamp_x0),
            autocast=(not args.no_amp),
        )
        # Map from [-1,1] to [0,1]
        x = (x.clamp(-1, 1) + 1) * 0.5
        all_imgs.append(x.cpu())

    imgs = torch.cat(all_imgs, dim=0)

    # Save a grid
    grid = make_grid(imgs, nrow=int(math.sqrt(n)) if int(math.sqrt(n))**2 == n else 8, padding=2)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_image(grid, args.out)
    print(f"Saved: {args.out}")

    # Optional: print any useful diffusion info if present in ckpt
    for k in ["T", "timesteps", "diffusion_steps", "cfg", "config"]:
        if isinstance(raw, dict) and k in raw:
            print(f"Note: checkpoint has key '{k}' (not used unless you wire it).")


if __name__ == "__main__":
    main()
