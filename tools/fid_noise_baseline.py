"""
Compute a "zero-skill" FID baseline: pure noise vs CIFAR-10 stats.

Usage example (Colab):

python tools/fid_noise_baseline.py \
  --fid-stats stats/cifar10_inception_train.npz \
  --n-images 10000 \
  --batch-size 250 \
  --seeds 0 \
  --device cuda \
  --out /content/drive/MyDrive/min-snr-noise-vs-stats-baseline/fid_noise_baseline.jsonl 
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablation_harness.eval.generative import _inception_activations, _fid_from_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-skill FID baseline: pure noise vs stats")

    p.add_argument(
        "--fid-stats",
        type=str,
        required=True,
        help="Path to .npz with reference mu/sigma (same one you use for eval).",
    )
    p.add_argument(
        "--n-images",
        type=int,
        default=50000,
        help="Total number of noise images to sample.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=250,
        help="Batch size for Inception forward pass.",
    )
    p.add_argument(
        "--height",
        type=int,
        default=32,
        help="Image height (CIFAR-10 = 32).",
    )
    p.add_argument(
        "--width",
        type=int,
        default=32,
        help="Image width (CIFAR-10 = 32).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: 'cuda' or 'cpu'.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[0],
        help="Random seeds to run. One FID will be computed per seed.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional JSONL output file to append results to.",
    )
    p.add_argument(
        "--noise-mode",
        type=str,
        default="uniform",
        choices=["uniform", "gaussian"],
        help="Distribution for noise in generator space [-1,1].",
    )
    return p.parse_args()


def make_noise_images(
    n: int,
    c: int,
    h: int,
    w: int,
    device: torch.device,
    mode: str = "uniform",
) -> torch.Tensor:
    """
    Create a batch of "generator outputs" in [-1,1].

    We sample directly in generator space since your UNet outputs live there.
    """
    if mode == "uniform":
        x = torch.empty(n, c, h, w, device=device).uniform_(-1.0, 1.0)
    elif mode == "gaussian":
        # Gaussian, then clamp to [-1,1] so it's still in valid pixel range.
        x = torch.randn(n, c, h, w, device=device)
        x = x.clamp(-1.0, 1.0)
    else:
        raise ValueError(f"Unknown noise mode: {mode}")

    return x


def noise_fid_once(
    fid_stats_path: Path,
    n_images: int,
    batch_size: int,
    h: int,
    w: int,
    device: torch.device,
    seed: int,
    noise_mode: str,
) -> float:
    # Set RNG for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load reference stats
    stats = np.load(str(fid_stats_path))
    mu_ref = stats["mu"]
    sigma_ref = stats["sigma"]

    # Generate noise images in [-1,1] in batches, compute Inception features
    feats_list = []
    n_done = 0

    while n_done < n_images:
        bs = min(batch_size, n_images - n_done)
        x_gen = make_noise_images(bs, 3, h, w, device=device, mode=noise_mode)  # [-1,1]

        # Map to [0,1] exactly like your real FID path:
        x_01 = (x_gen.clamp(-1.0, 1.0) + 1.0) / 2.0  # [-1,1] -> [0,1]

        feats = _inception_activations(x_01, device)  # np.ndarray [bs, D]
        feats_list.append(feats)

        n_done += bs

    feats_all = np.concatenate(feats_list, axis=0)
    assert feats_all.shape[0] == n_images

    mu_noise = feats_all.mean(axis=0)
    sigma_noise = np.cov(feats_all, rowvar=False)

    fid = _fid_from_stats(mu_noise, sigma_noise, mu_ref, sigma_ref)
    return float(fid)


def main() -> None:
    args = parse_args()

    fid_stats_path = Path(args.fid_stats)
    if not fid_stats_path.exists():
        raise FileNotFoundError(f"FID stats file not found: {fid_stats_path}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"FID stats: {fid_stats_path}")
    print(f"Noise mode: {args.noise_mode}")
    print(f"Images: {args.n_images}, batch_size: {args.batch_size}")
    print(f"Seeds: {args.seeds}")

    out_path = Path(args.out) if args.out else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fids = []

    for seed in args.seeds:
        fid = noise_fid_once(
            fid_stats_path=fid_stats_path,
            n_images=args.n_images,
            batch_size=args.batch_size,
            h=args.height,
            w=args.width,
            device=device,
            seed=seed,
            noise_mode=args.noise_mode,
        )
        fids.append(fid)
        msg = f"seed={seed}  n={args.n_images}  FID(noise, stats)={fid:.3f}"
        print(msg)

        if out_path is not None:
            rec = {
                "seed": seed,
                "n_images": args.n_images,
                "batch_size": args.batch_size,
                "height": args.height,
                "width": args.width,
                "device": str(device),
                "noise_mode": args.noise_mode,
                "fid": fid,
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    if len(fids) > 1:
        mean = float(np.mean(fids))
        std = float(np.std(fids))
        print(f"\nmean FID(noise, stats) = {mean:.3f} ± {std:.3f} (over {len(fids)} seeds)")


if __name__ == "__main__":
    main()