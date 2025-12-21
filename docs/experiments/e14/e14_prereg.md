# Prereg: E14 — Seed sensitivity (min-snr, linear, NFE=50, 3 seeds)

**ID:** E14  
**Question:** How much do random seeds move our headline metrics under the minsnr config?  
**Motivation:** See the min-snr seed spread in comparasion.

## Config 
- **Seeds:** `[1077, 11, 1]` (3 runs)
- **Data:** CIFAR-10 full, `batch_size=64`, shuffle, workers=2
- **Model:** `unet_cifar32`, `base_channels=64`
- **Diffusion:** `beta_schedule=linear`
- **Loss:** `weighting=minsnr`
- **Train:** `total_steps=10_000`, `amp=true`, `grad_clip=1.0`
- **EMA:** enabled, decay=0.999
- **Eval (key):**
  - **KID:** every 2500 steps, DDIM, **NFE=50**, `n_samples=5000`, `repeats=20`
  - **FID milestone:** every 2500 steps, DDPM, **NFE=50**, `n_samples=5000`, stats = `stats/cifar10_inception_train.npz`
  - **Grid:** at 10k, DDIM, NFE=50, 36 samples
  - **Recon:** every 2000, 4 images, metrics (MSE/PSNR)

## Outcomes
### Primary endpoints (seed variability)
At **step = 10,000** (and also at 2.5k/5k/7.5k):
- `val/fid` (from `fid_milestone`)
- KID mean (the logged KID aggregate for that eval)

Report across seeds:
- mean, sample std (n=3), min/max (range)

### Secondary
- Wall-clock: total runtime, sec/step, eval time share 
- Training loss curve shape + final loss
- Recon MSE/PSNR at fixed t values (sanity, not a decision metric)

## Hypothesis (directional)
Seed-to-seed spread exists but should be smaller than the deltas we’re chasing in later interventions (schedule / reweighting). This run is to measure the spread, not to win. Can also be compared to baseline.

## Execution plan
1. Run E14 with three seeds (one run per seed).
2. After completion, summarize the endpoints above + attach plots/tables.

## Decision use
- Use the measured seed std/range as the noise floor when judging later experiment deltas.
- If spread is large, consider: more seeds, or stabilizing eval (e.g., fewer repeats but more seeds, or vice versa), before concluding anything from small FID/KID changes.
