# e21 — Preregistration: LR sensitivity probe (Min-SNR γ=5)

## Experiment ID
**e21** — Hyperparameter sensitivity probe: learning-rate sweep for Min-SNR.

**Goal:** test whether the (e20-style) Min-SNR setup is sensitive to small LR changes.

---

## Hypotheses
- **H1 (robustness):** modest LR shifts around 1e-4 do not materially change quality by 10k steps (differences are small / within typical noise).
- **H2 (directional):** if there is a trend, slightly lower LR improves stability/EMA and may improve FID/KID, higher LR may speed early loss but can worsen final sample quality.

---

## Design
### Independent variable
`optim.lr` (three variants):
- **lrA:** 5.0e-5
- **lrB:** 1.3e-4
- **lrC:** 2.0e-4

### Controls (held constant)
Everything else is identical to the provided YAML:
- CIFAR-10, batch_size=64, cosine beta schedule
- model: unet_cifar32 (base_channels=64)
- EMA: enabled, decay=0.999
- steps: 10,000, grad_clip=1.0, AMP=true
- loss: Min-SNR weighting, gamma=5.0
- eval cadence: grid every 1k; recon every 2k; KID/FID-milestone every 2.5k; NFE=20 (DDIM)
- seed: **11** (same seed for all three runs to isolate LR effects)

---

## Run plan (exact configs)
Run these three YAMLs:
- `e21_lr_5e-5_min-snr.yaml`
- `e21_lr_1.3e-4_min-snr.yaml`
- `e21_lr_2e-4_min-snr.yaml`

---

## Endpoints
### Primary endpoint
- **FID** (`val/fid`) from **FID-milestone** evaluations (every 2500 steps), with the primary comparison at **step 10,000**.

### Secondary endpoints
- **KID mean** from KID evals (n_samples=5000, repeats=5) at the same milestones.
- Train dynamics: `train/loss` vs step.
- Recon: MSE / PSNR at fixed `t_values` at each recon checkpoint (every 2000).
- Qualitative: grid images (every 1000).

---

## Success criteria / interpretation rules (set before viewing results)
Define meaningful change as:
- **Meaningful ΔFID:** ≥ **1.0** FID at **step 10k**, *or* consistent advantage across ≥2 milestones (not a one-off).

Classification:
- If no LR variant beats the baseline regime by ≥1.0 FID: conclude insensitive within this LR range for this setup.
- If one LR wins by the threshold and looks stable: adopt it for the next holdout / seed sweep stage.

---

## Stop rules
- If loss becomes **NaN/Inf** or training crashes → mark that run **FAILED** and stop it.
- If sample grids show clear collapse and FID is catastrophically worse at ≥2 consecutive milestones → stop early, record as unstable at this LR.

---

## Analysis plan (pre-declared plots)
Generate the same plots for all three runs:
1. `val/fid` vs step (milestones)
2. `val/kid_mean` vs step 
3. `train/loss` vs step
4. Recon profile: PSNR and/or MSE vs `t` at final recon checkpoint
5. Grid montage at the last available step (prefer step 10k)
---

## Notes
This prereg is intentionally minimal. it isolates LR sensitivity only (no seed sweep, no other hyperparameters).
