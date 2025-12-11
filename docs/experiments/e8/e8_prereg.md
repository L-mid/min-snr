# E8 – Min-SNR γ sweep (mini, 5k, bc64)

**ID:** E8  
**Study:** MS1_min_snr  
**Goal:** Pick a “good enough” Min-SNR γ ∈ {1, 3, 5} for later full 50k runs, using tiny 5k pilots with base_channels=64.

**Configs:**
- E8a (γ=1): `configs/study/MS1_min_snr/e8/e8a_minsnr_linear_gamma1_5k_bc64.yaml`
- E8b (γ=3): `configs/study/MS1_min_snr/e8/e8b_minsnr_linear_gamma3_5k_bc64.yaml`
- E8c (γ=5): `configs/study/MS1_min_snr/e8/e8c_minsnr_linear_gamma5_5k_bc64.yaml`

---

## Design (short)

- **Data:** CIFAR-10, `batch_size=4`, standard split, shuffle.
- **Model:** `unet_cifar32`, `base_channels=64`, defaults otherwise.
- **Diffusion:** DDPM, `beta_schedule="linear"`.
- **Loss:** ε-MSE with Min-SNR.
  - E8a: `gamma=1.0`
  - E8b: `gamma=3.0`
  - E8c: `gamma=5.0`
- **Optim:** Adam, `lr=1e-4`, `grad_clip=1.0`, `amp=true`.
- **Train:** `total_steps=5000`, `seed=1077`, EMA `decay=0.999`.
- **Eval:**
  - FID milestones: every 2000 steps, `ddpm`, `nfe=50`, `n_samples=5000`.
  - Final FID @5k: `ddim`, `nfe=50`, `n_samples=10000`.
  - Grid: @5k, `ddim`, `nfe=20`, `n_samples=36`, save images.
- **Curvature:** Hutchinson enabled (diagnostic only).

---

## Primary metric & comparison

- **Main metric:** validation FID vs step (EMA model).
- For each γ, record:
  - `FID_2k`, `FID_4k`, `FID_5k`
  - `FID_min(0–5k)` (min FID over run)

---

## Scope / caveats

- Single seed, 5k steps → **pilot only**, not a final comparison vs baselines.
- Only used to choose a candidate γ; small FID gaps (<5–10%) are treated as noise.