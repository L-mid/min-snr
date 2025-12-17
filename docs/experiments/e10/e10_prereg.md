# E10 — Sampler sanity (DDPM vs DDIM @ NFE=20, fixed Min-SNR)

**ID:** E10  
**Study:** `min_snr` (`study/v1`)  
**Goal:** Confirm Min-SNR does not break sampling, and determine whether very high FID is due to training limitations vs an eval/sampler bug.

---

## Runs

- **E10a:** `e10a-minsnr-fixed-ddpm-nfe20-10k-bc64`  
  Eval sampler: **DDPM**, NFE=20

- **E10b:** `e10b-minsnr-fixed-ddim-nfe20-10k-bc64`  
  Eval sampler: **DDIM**, NFE=20

**Everything else identical** (CIFAR-10, bs=4, UNetCifar32 bc64, EMA 0.999, cosine β, K=1000, 10k steps, Adam 1e-4, AMP, Min-SNR γ=5, deterministic=true).

---

## Questions

1) Does sampling behave correctly (no NaNs/divergence; reasonable grids)?  
2) Why is FID bad: underfitting/capacity/batch-size vs bug (sampler/t-schedule/seed/clamp)?

---

## Hypotheses

- If **DDPM ≈ DDIM** (both bad similarly): likely **training-side** (underfit, bs=4 noise, capacity/steps).
- If **DDIM << DDPM** or looks visually broken: likely **DDIM/eval path** bug.

---

## Metrics / Artifacts

- **Primary metric:** `val/fid`
- **Artifacts:** `grid.png` every 2000 steps (36 samples)
- **FID milestone:** every 2000 steps, `n_samples=5000`
- **Final FID:** end of run, `n_samples=10000`
- **Recon** every 2000 steps, `metrics: ["mse", "psnr"]`
---

## Interpretation buckets

- **Training-limited:** DDPM and DDIM grids both look like noise/weak structure; FID both huge → undertraining/capacity/batch-size.
- **Sampler/eval bug:** DDIM grids obviously wrong vs DDPM; FID much worse → inspect DDIM math / t_schedule / seeding.
- **Seed artifact:** identical grids/means across milestones → seed reuse/other
