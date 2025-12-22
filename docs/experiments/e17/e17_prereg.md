# Prereg: E17 — Seed sensitivity (basline, cosine, NFE=25, 3 seeds)

**ID:** E17  
**Question:** How much do random seeds change under cosine (baseline)? 
**Motivation:** Cosine previously lead to qualitivly better results, does this uphold vs multipule seeds? (Will also test for min-snr under cosine after)
 
## Config 
- **Seeds:** `[1077, 11, 1]` (3 runs)
- **Data:** CIFAR-10 full, `batch_size=64`, shuffle, workers=2
- **Model:** `unet_cifar32`, `base_channels=64`
- **Diffusion:** `beta_schedule=cosine`
- **Loss:** `weighting=constant`
- **Train:** `total_steps=10_000`, `amp=true`, `grad_clip=1.0`
- **EMA:** enabled, decay=0.999
- **Eval (key):**
  - **KID:** every 2500 steps, DDIM, **NFE=25**, `n_samples=5000`, `repeats=20`
  - **FID milestone:** every 2500 steps, DDPM, **NFE=25**, `n_samples=5000`, stats = `stats/cifar10_inception_train.npz`
  - **Grid:** at 10k, DDIM, NFE=25, 36 samples, also every 1000 steps.
  - **Recon:** every 2000, 4 images, metrics (MSE/PSNR)

## Outcomes
### Primary endpoints (seed variability)
At **step = 10,000** (and also at 2.5k/5k/7.5k):
- `val/fid` (from `fid_milestone`)
- KID mean (the logged KID aggregate for that eval)

Report across seeds:
- mean, sample std (n=3), min/max (range)

### Secondary
- Wall-clock: total runtime, sec/step, eval time share (compare this to baseline nfe=25)
- Training loss curve shape + final loss
- Sample evolution over every 1000 steps compared to FID/KID.
- Recon MSE/PSNR at fixed t values (sanity, not a decision metric)

## Hypothesis (directional)
Seed-to-seed spread very small and stable like in baseline, even with cosine enabled, and wallclock similar/same for similar or better results (lower FID, KID, samples).

## Execution plan
1. Run E17 with three seeds (one run per seed).
2. After completion, summarize the endpoints above + attach plots/tables.

## Decision use
- Use the measured seed std/range as the noise floor when judging later experiment deltas and the practicality of single seed runs while still testing.
- If spread is large, consider: more seeds, or stabilizing eval (e.g., fewer repeats but more seeds, or vice versa), before concluding anything from small FID/KID changes. 
- If cosine a clear and stable improvement (>-3 FID), use as a part of "best baseline config" later. 
