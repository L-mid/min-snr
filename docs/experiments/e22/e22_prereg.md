# Pre-registration — e22: Holdout confirm @ NFE=35 (Min-SNR γ=5, cosine)

## Goal
Validate that the **best Min-SNR config (γ=5, cosine β schedule)** remains strong on a holdout-style seed set when evaluated with a higher DDIM sampling budget (NFE=35), and quantify the quality ↔ wall-clock tradeoff vs the prior NFE=20 holdout (e20).

## Hypotheses
- **H1 (quality):** Using DDIM NFE=35 for evaluation (grid/KID/FID-milestone) will produce lower FID than NFE=20 on the same trained checkpoints, and performance remains stable across seeds.
- **H2 (ranking stability):** The Min-SNR γ=5 + cosine setup does not collapse at NFE=35; seed variance remains comparable to prior holdout behavior.

## Experiment identity
- **Study:** `min_snr`
- **Experiment ID:** `e22`
- **Description:** Holdout confirmation run with eval NFE raised to 35.
- **Seeds:** `[2222, 3333, 4444]`
- **Training budget:** 10,000 steps

## Config 
Everything matches the best Min-SNR config I've been using, except evaluation NFE.

### Data
- Dataset: CIFAR-10 (full)
- Batch size: 64
- Num workers: 2
- Shuffle: true

### Model
- `unet_cifar32`
- base_channels: 64

### Optim / Train
- Optimizer: Adam
- LR: `1e-4`
- total_steps: 10,000
- grad_clip: 1.0
- AMP: true

### EMA
- enabled: true
- decay: 0.999

### Diffusion
- β schedule: cosine

### Loss
- weighting: Min-SNR
- γ: 5.0

## Evaluation plan
Sampler for eval tasks: **DDIM**.

### Schedules
- **Grid:** every 1000 steps, n_samples=36, NFE=35
- **KID:** every 2500 steps, n_samples=5000, repeats=5, subset_size=100, NFE=35
- **FID milestone:** every 2500 steps, n_samples=5000, **NFE=35**, stats=`stats/cifar10_inception_train.npz`
- **Final:** disabled
- **Recon diagnostics:** every 2000 steps, t fixed list `[10, 200, 500, 700, 900, 999]`, metrics `[mse, psnr]`, save_images=true

## Primary outcome
- **Primary metric:** `val/fid` from **FID milestone** at steps 2500 / 5000 / 7500 / 10000
- Report:
  - per-seed values
  - mean ± std across seeds

## Secondary outcomes
- KID (mean ± std across seeds) at the same milestones
- Recon metrics (MSE/PSNR) at fixed timesteps (sanity/diagnostic)
- Wall-clock:
  - total runtime per seed run
  - evaluation runtime per milestone (KID/FID), if available
  - overall quality per time vs e20

## Comparison baseline
- Compare against e20 holdout @ NFE=20 (same training recipe, different eval NFE).
- Purpose: quantify FID/KID improvements vs added eval cost.


- **Pass (confirm):** At 10k steps, mean FID across seeds is **not worse than e20 by more than +2.0 FID**, and ideally improves (expected due to higher NFE).
- **Strong confirm:** mean FID improves by a visible margin and seed variance stays comparable.


## Planned plots 
1. **FID vs step** (per-seed + mean), overlay e20 mean as reference
2. **KID vs step** (per-seed + mean), overlay e20 mean
3. **Wall-clock vs step** (show eval spikes)
4. **FID vs wall-clock** (frontier: e20 vs e22)
5. **Grid samples @ 10k** for each seed 


