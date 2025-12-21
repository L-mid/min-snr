# Prereg: E16 — Seed sensitivity (min-snr, linear, NFE=25, 3 seeds)

**ID:** E16  
**Question:** How much do random seeds uphold under lower sampling nfe similary to baseline without metric and grid quality collapse?  
**Motivation:** See how lower nfe changes seed spread of metrics compared to A) min-snr nfe=50, B) baseline nfe=25.

## Config 
- **Seeds:** `[1077, 11, 1]` (3 runs)
- **Data:** CIFAR-10 full, `batch_size=64`, shuffle, workers=2
- **Model:** `unet_cifar32`, `base_channels=64`
- **Diffusion:** `beta_schedule=linear`
- **Loss:** `weighting=min-snr`
- **Train:** `total_steps=10_000`, `amp=true`, `grad_clip=1.0`
- **EMA:** enabled, decay=0.999
- **Eval (key):**
  - **KID:** every 2500 steps, DDIM, **NFE=25**, `n_samples=5000`, `repeats=20`
  - **FID milestone:** every 2500 steps, DDPM, **NFE=25**, `n_samples=5000`, stats = `stats/cifar10_inception_train.npz`
  - **Grid:** at 10k, DDIM, NFE=25, 36 samples
  - **Recon:** every 2000, 4 images, metrics (MSE/PSNR)

## Outcomes
### Primary endpoints (seed variability)
At **step = 10,000** (and also at 2.5k/5k/7.5k):
- `val/fid` (from `fid_milestone`)
- KID mean (the logged KID aggregate for that eval)

Report across seeds:
- mean, sample std (n=3), min/max (range)

### Secondary
- Wall-clock: total runtime, sec/step, eval time share (compare this to baseline)
- Training loss curve shape + final loss
- Recon MSE/PSNR at fixed t values (sanity, not a decision metric)

## Hypothesis (directional)
Seed-to-seed spread very small and stable like in baseline (min-snr does not collapse convergance), and low nfe brings down eval time for extremely similar metric results. This run is to measure the spread, not to win, and show that nfe=25 is comparable to nfe=50 in this regime. 

## Execution plan
1. Run E15 with three seeds (one run per seed).
2. After completion, summarize the endpoints above + attach plots/tables.
3. Examine the spread nfe causes, and degredation of metric quality at lower nfe. None/little effect is a valid (and desireable) result.

## Decision use
- Use the measured seed std/range as the noise floor when judging later experiment deltas.
- If spread is large, consider: more seeds, or stabilizing eval (e.g., fewer repeats but more seeds, or vice versa), before concluding anything from small FID/KID changes.
