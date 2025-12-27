# Prereg: E19 — Hold-out baseline (best vanilla config, 3 new seeds)

**ID:** E20  
**Question:** What is the performance + seed variability of my best min-snr config on the 3 fresh hold-out seeds?  
**Motivation:** Lock a clean, trustworthy reference point (noise floor + runtime) to judge later deltas against, no reusing tuning seeds.

## Config (vanilla baseline, locked)
- **Seeds (hold-out):** `[2222, 3333, 4444]` (3 runs; same as e20)
- **Data:** CIFAR-10 full, `batch_size=64`, shuffle, workers=2
- **Model:** `unet_cifar32`, `base_channels=64`
- **Diffusion:** `beta_schedule=cosine`
- **Loss:** `weighting=min-snr` 
- **Train:** `total_steps=10_000`, `amp=true`, `grad_clip=1.0`
- **EMA:** enabled, decay=0.999
- **Determinism:** `deterministic=true`

## Eval (key, locked + consistent)
- **KID:** every 2500 steps, **DDIM**, **NFE=20**, `n_samples=5000`, `repeats=5`
- **FID milestone:** every 2500 steps, **DDIM** (testing how DDIM changes results here), **NFE=20**, `n_samples=5000`, stats = `stats/cifar10_inception_train.npz`
- **Grid:** every 1000 steps + at 10k, **DDIM**, NFE=20, 36 samples
- **Recon:** every 2000, 4 images, metrics (MSE/PSNR)

## Outcomes
### Primary endpoints (hold-out baseline + seed variability)
At **step = 10,000** (and also at 2.5k/5k/7.5k):
- `val/fid` (from `fid_milestone`, DDIM)
- KID mean (logged KID aggregate, DDIM)

Report across seeds:
- mean, sample std (n=3), min/max (range)

### Secondary
- Wall-clock: total runtime, sec/step, eval time share
- Training loss curves + final loss
- Sample evolution every 1000 steps (grids) vs FID/KID
- Recon MSE/PSNR at fixed t values 

## Hypothesis 
- Hold-out seed spread is similar to E19’s spread despite being min-snr, and min-snr wins on final fid by 2-3 points.




