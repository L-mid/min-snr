# W1-E7: Baseline DDPM – Steps vs Capacity

**Variants**

- **E7a** – Baseline reference rerun (10k steps, base_channels=32)  
- **E7b** – Longer baseline (50k steps, base_channels=32)  
- **E7c** – Wider baseline (10k steps, base_channels=64)

---

## Question

Why is FID stuck ≈ 200?

Is it mainly:

1. Too few training steps (under-training)?  
2. Too small a UNet (under-capacity)?  
3. Or something else (metric / normalization / sampler / implementation)?

E7a/b/c probe (1) and (2) while holding everything else fixed.

---

## Configs

**Planned paths**

- `configs/study/MS1_min_snr/e7a/e7a_baseline_linear_10k.yaml`
- `configs/study/MS1_min_snr/e7b/e7b_baseline_linear_50k.yaml`
- `configs/study/MS1_min_snr/e7c/e7c_baseline_linear_10k_bc64.yaml`

**Shared setup (all)**

- Dataset: CIFAR-10 train  
- Model: `unet_cifar32` (GN, dropout=0.1)  
- Loss: vanilla ε-MSE (`loss.weighting = "constant"`)  
- Diffusion: linear β, default T  
- Batch size: 4, lr=1e-4 (Adam)  
- EMA: 0.999  
- Curvature: Hutchinson trace, 16 probes  
- FID: DDPM, 50 NFE, CIFAR-10 train stats  
- FID milestones: every 2k steps; final FID at end

**Variant knobs**

- **E7a**: `base_channels = 32`, `total_steps = 10_000`  
- **E7b**: `base_channels = 32`, `total_steps = 50_000`  
- **E7c**: `base_channels = 64`, `total_steps = 10_000`

---

## Hypotheses

- **Under-training hypothesis**  
  If steps are the bottleneck, **E7b** should get clearly better best-FID than **E7a**.

- **Under-capacity hypothesis**  
  If capacity is the bottleneck, **E7c** should get clearly better best-FID than **E7a** at 10k steps.

- **Elsewhere hypothesis**  
  If **E7b ≈ E7a** and **E7c ≈ E7a** in FID, then the main problem is likely not just steps or capacity (suspect FID pipeline, normalization, sampler, or other implementation details).

---

## Primary comparisons

1. **Sanity / regression check**

   - E7a vs E5 vs E6:
     - `train/loss` vs steps  
     - `curvature/hutch_trace_mean` vs steps  
     - `val/fid` at 2k,4k,6k,8k,10k  

2. **Steps effect**

   - E7b vs E7a: FID vs steps, best FID over run.

3. **Capacity effect**

   - E7c vs E7a: FID vs steps (0–10k), best FID at ≤10k.

4. **Steps vs capacity**

   - E7b vs E7c: which gives the better best-FID per T4-style budget.

---

## Definition of Done

- All three runs complete with:
  - Loss, curvature, and FID logged.  
  - Overlay plots for:
    - Loss vs steps (E5/E6/E7a/b/c)  
    - Curvature vs steps (where available)  
    - FID vs steps (may use multiple clearer overlays).

- Interpretation:
  - **E7b ≪ E7a** → under-training is a major bottleneck.  
  - **E7c ≪ E7a** → under-capacity is a major bottleneck.  
  - **Both improve** → both matter; pick a sweet-spot config.  
  - **Neither improves** → investigate FID / normalization / sampler / implementation.