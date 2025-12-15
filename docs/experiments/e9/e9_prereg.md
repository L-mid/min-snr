# MS1 – e9: Scheduler interaction (cosine vs Min-SNR)

## Question

Does Min-SNR loss reweighting (γ = 5) behave fine under a cosine β schedule on CIFAR-10, or does it interact with the scheduler in a pathological way compared to vanilla ε-MSE?

## Hypothesis

- H1 (sanity): Under a cosine β schedule, Min-SNR (γ = 5) trains stably and produces samples with FID in the same rough band as a vanilla ε-MSE baseline (no obvious collapse or divergence).
- H2 (soft): The qualitative training behaviour (loss curves, FID vs steps, curvature trends) for Min-SNR under cosine is similar to the linear-β case and does not show obviously worse behaviour than vanilla under cosine. (particulary looking for lower FID, even if marginal).

Primary goal: pathology check, not beat baseline, and **check if FID degedation persists/indeed worsens with cosine**.

---

## Experimental Design

- Dataset: CIFAR-10 (train set for training; standard FID stats file for eval).
- Model: `unet_cifar32`, `base_channels = 64`.
- Diffusion:
  - `beta_schedule = "cosine"` for both runs.
  - Same T and other diffusion hyperparameters as in earlier MS1 runs.
- Optimizer:
  - Adam, `lr = 1e-4`.
- Training:
  - `total_steps = 10_000` (short run)
  - Batch size = 4, AMP on, grad clip = 1.0
  - EMA enabled, `decay = 0.999`
- Seed: `1077` for both runs.
- Curvature:
  - Hutchinson trace on, `probes = 16`
  - Logged under `curvature/hutch/*`.

This is explicitly a paired comparison: keep everything identical except the loss weighting under cosine.

---

## Conditions / Configs

- **e9a – Vanilla ε-MSE baseline, cosine β**
  - Config:  
    `configs/study/MS1_min_snr/e9/e9a_baseline_cosine_10k.yaml`
  - Key differences:
    - `loss.weighting: "uniform"` (or whatever the vanilla ε-MSE setting is in this codebase)
    - `beta_schedule: "cosine"`
    - 10k steps, logging every 500 steps.

- **e9b – Min-SNR (γ = 5), cosine β**
  - Config:  
    `configs/study/MS1_min_snr/e9/e9b_minsnr_cosine_gamma5_10k.yaml`
  - Key differences from e9a:
    - `loss.weighting: "minsnr"`
    - `loss.minsnr_gamma: 5.0`
    - Same cosine schedule, same training budget.

---

## Metrics & Logging

- **Primary metric**
  - `val/fid` using DDPM sampler, `nfe = 50`, `n_samples = 5000` at milestones and `10000` at final.
  - Milestones: every 2000 steps (2k, 4k, 6k, 8k, 10k).
  - Compare:
    - Best FID over 10k steps per run.
    - FID trajectory shape (does Min-SNR blow up, plateau, or look similar).
    - wishing best of luck for FID!

---

## Analysis Plan

1. **FID vs steps overlay (e9a vs e9b)**
   - Plot FID @ {2k, 4k, 6k, 8k, 10k} vs steps.
   - Check:
     - Does Min-SNR track roughly in the same band as vanilla?
     - Any sudden spikes or divergence?

2. **Loss & curvature behaviour**
   - Plot train loss vs steps (overlay e9a/e9b).
   - Plot curvature metrics (Hutchinson trace) vs steps for both runs.
   - Looking for:
     - Exploding curvature! unique to Min-SNR under cosine.
     - Strongly different curvature profile vs linear-β Min-SNR runs (e7/e8 context).

3. **Qualitative samples**
   - Compare final grids:
     - Are Min-SNR samples clearly “broken” (e.g. high noise, washed out, mode collapse) relative to vanilla under cosine?
     - If both are similarly OK/mediocre, treating that as no obvious scheduler pathology in particular.

---

## Success / Stop Criteria

- **Success (for this e9 check):**
  - Training completes to 10k steps for both runs with no instabilities.
  - Min-SNR FID is **not catastrophically worse** than vanilla (e.g. within ~20–30% relative, or at least same qualitative scale).
  - No clear, Min-SNR-only pathologies in loss, curvature, or images.

- **Failure / pathology flag:**
  - Min-SNR diverges (loss NaN/inf) while vanilla is fine.
  - Min-SNR FID is dramatically worse (e.g. ~2× or more) than vanilla under the same cosine schedule.
  - Qualitative samples show obvious collapse artefacts unique to Min-SNR.
  - Curvature / SNR stats show extreme behaviour for Min-SNR not present in vanilla.

If a pathology is detected, e10+ may focus on isolating "cosine × Min-SNR" interaction (e.g. by adjusting γ, cap strategy, or SNR definition). If not, great!

