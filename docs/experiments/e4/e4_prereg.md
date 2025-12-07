# [W1-E4] – Min-SNR weight normalization (mean-normalized weights)

## Question

In the small CIFAR-10 DDPM setup, how much of Min-SNR’s effect comes from its shape over timesteps versus its overall scale?

Concretely: if we take the Min-SNR loss weights used in W1-E2 and normalize them to have mean 1 across timesteps (so the average effective learning rate is preserved), does training behaviour or FID change in a systematic way?

---


## Hypotheses

1. **H1 – Scale vs shape:**  
   If Min-SNR’s main benefit in this regime is just “cranking up” the effective LR on a subset of timesteps, then mean-normalizing to 1 should reduce or remove any E2 gains relative to baseline.

2. **H2 – Shape matters:**  
   If Min-SNR’s *shape* over t (focusing high-SNR or mid-SNR timesteps) is actually the important part, then:
   - E4 should look closer to E2 than to the plain ε-MSE baseline in terms of FID vs steps.
   - Gradient energy over t should still be redistributed toward the Min-SNR focus region, even after normalization.

3. **H3 – No LR blow-ups:**  
   After normalization:
   - The mean gradient norm per step should be within a small factor (e.g. ≤ 2×) of the vanilla ε loss.
   - There should be no sustained explosion in gradient norms or NaN/Inf issues.

## Configuration

- **Config path (planned):**  
  `configs/study/MS1_min_snr/e4/e4_minsnr_weightnorm.yaml`

- **Key params (same as W1-E2 unless noted):**
  - Dataset: CIFAR-10, 32×32, full train set.
  - Model: `UNet_CIFAR32` (same width/depth/attention as E1/E2).
  - Diffusion:
    - Linear β schedule, T and ᾱ_t exactly as in E1/E2.
  - Optimizer:
    - Adam or AdamW (same lr, betas, weight decay as E1/E2).
  - Loss:
    - Type: ε-prediction.
    - Weighting: `minsnr_norm`.
    - γ: same as W1-E2 Min-SNR config.
  - Training:
    - Steps: 50k.
    - Batch size: same as E1/E2.
  - Sampling / eval:
    - Sampler: DDIM, NFE=50.
    - FID: at {2k, 10k, 25k, 50k} steps on ≥10k samples (same as E3).

---


## Metrics

**Primary:**

- FID@10k, 25k, 50k (NFE=50), comparing:
  - E1: baseline ε-MSE, linear β, 50k.
  - E2: Min-SNR (raw weights), linear β, 50k.
  - E4: Min-SNR (mean-normalized weights), linear β, 50k.

**Secondary / diagnostic:**

- **Weight curves:**  
  - `mins_snr_curve/weights_raw[t]` vs `mins_snr_curve/weights_norm[t]`
  - `mins_snr_curve/weights_norm_mean` (should be ≈ 1.0).
- **Gradient diagnostics (logged occasionally, e.g. every 500 or 1k steps):**
  - `grads/global_norm/eps_loss`, `grads/global_norm/minsnr_raw`, `grads/global_norm/minsnr_norm`.
  - Optionally: gradient norm per-t bucket (coarse bins like low/mid/high t).
- Training loss vs steps:
  - `train/loss` and a stratified loss over t buckets.

---

## Planned analyses / plots

At minimum:

1. **FID vs steps overlay (E1, E2, E4)**  
   - X: training steps (up to 50k).  
   - Y (left): FID.  
   - Curves: E1, E2, E4 (NFE=50).  
   - Does E4 sit with E3 or fall back toward E1?

2. **Raw vs normalized Min-SNR weights (shape vs scale)**  
   - X: t / T in [0, 1].  
   - Y: weight(t).  
   - Curves: raw Min-SNR (E2) vs normalized Min-SNR (E4).  
   - Highlight: identical shape, different overall scale; verify mean≈1 for E4.

3. **“Effective LR” / gradient-energy plot**  
   - Option A (cheap): plot `global_grad_norm` for:
     - ε baseline,
     - raw Min-SNR,
     - normalized Min-SNR,
     versus training steps.
   - Option B (fancier): heatmap with:
     - X: timestep buckets,
     - Y: training iteration,
     - Color: relative contribution of each t-bucket to total loss.  
   - Goal: show whether E4 still shifts gradient energy to the same region in t as E2.

4. **Per-t loss or error profile at a fixed checkpoint**  
   - Pick a representative step (e.g. 10k or 50k).  
   - Compute mean ε-MSE vs t for E1, E2, E4 on a held-out batch.  
   - Plot X=t/T, Y=ε-MSE(t).  
   - Goal: see where in t each model is actually “good”.

---

## Risks / interpretation notes

- If all configs are massively undertrained, improvements might be noisy. The main value then is **diagnostic** (how gradient energy moves) rather than absolute FID.
- If gradient norms still explode even after mean-normalization, that suggests Min-SNR’s shape itself is too peaky for this tiny model / LR, and we may need:
  - gradient clipping, or
  - an explicit cap on `max_weight`.