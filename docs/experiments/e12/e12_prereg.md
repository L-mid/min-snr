# E12 – Batch size vs steps (constant images) playground

**ID:** E12  
**Study:** min_snr  
**Date:** 2025-12-17  
**Goal:** Test whether “bigger batch + fewer optimizer steps” is a good **time-saving scouting** strategy (wall clock) *without* destroying sample quality. Use results to decide what to promote into the next full scan.

---

## Question

At fixed total training images (≈40k), how does increasing **batch size** and reducing **steps** affect:

1) **wall clock time** (primary)  
2) sample quality (**KID/FID**)  
3) recon metrics + qualitative grids

---

## Hypotheses

- **H1 (speed):** Larger batch runs (bs64/bs128) reduce wall clock substantially vs bs4 for the same images-seen budget.
- **H2 (quality):** Larger batch + fewer steps may hurt quality (worse KID/FID), because the optimizer gets fewer updates.
- **H3 (sweet spot):** bs64/steps625 is the best tradeoff; bs128/steps313 is most likely to degrade.

---

## Design

**Fixed across runs**
- Dataset: CIFAR-10 (train split for real features as configured)
- Model: `unet_cifar32`, `base_channels=64`
- Diffusion: `beta_schedule=cosine`
- Loss: `weighting=minsnr`, `minsnr_gamma=5`
- Optim: Adam, lr=1e-4, AMP on, grad_clip=1.0
- EMA: enabled, decay=0.999
- Sampler for grids/KID: DDIM, NFE=10

**Sweep (constant images ≈ 40k)**
- **E12a:** batch_size=4,  total_steps=10_000  → 40,000 images
- **E12b:** batch_size=64, total_steps=600     → ~40,000 images (rounded)
- **E12c:** batch_size=128,total_steps=320     → ~40,064 images (rounded)

---

## Runs / Configs

- E12a: `configs/study/MS1_min_snr/e12/e12a_playground_cosine_g5_bs4_steps10k_bc64.yaml`
- E12b: `configs/study/MS1_min_snr/e12/e12b_playground_cosine_g5_bs64_steps625_bc64.yaml`
- E12c: `configs/study/MS1_min_snr/e12/e12c_playground_cosine_g5_bs128_steps313_bc64.yaml`

---

## Metrics

**Primary**
- **Wall clock time** per run (total runtime) and derived throughput:
  - seconds / 40k images
  - images / second

**Secondary (quality)**
- **KID**: n_samples=2000, repeats=10, subset_size=100 (trend across checkpoints)
- **FID milestone**: n_samples=5000, DDPM NFE=20 at the same checkpoints
- **Recon**: MSE + PSNR at fixed timesteps (0, 10, 500, 900)
- **Qual**: grids (DDIM NFE=10)

---

## Eval cadence

Aim for ~4 checkpoints per run:
- E12a: every 2500 steps
- E12b: every 150 steps
- E12c: every 80 steps

(Each checkpoint runs: grid + recon + KID + FID milestone)

---

## Success criteria (promotion rule)

Prefer the fastest run that does not show a clear quality collapse:

Promote a setting if it achieves:
- ≥ **30% faster** wall clock than E12a and
- KID trend is not dramatically worse than E12a (no obvious divergence / instability) and
- FID milestone is not catastrophically worse (rule of thumb: within ~+15 absolute of E12a at the final checkpoint)

If bs128 is unstable / clearly worse, ignore and choose between bs4 vs bs64.

---

## Commands 
```bash
python -m ablation_harness.cli run --config configs/study/MS1_min_snr/e12/e12a_playground_cosine_g5_bs4_steps10k_bc64.yaml
python -m ablation_harness.cli run --config configs/study/MS1_min_snr/e12/e12b_playground_cosine_g5_bs64_steps625_bc64.yaml
python -m ablation_harness.cli run --config configs/study/MS1_min_snr/e12/e12c_playground_cosine_g5_bs128_steps313_bc64.yaml
