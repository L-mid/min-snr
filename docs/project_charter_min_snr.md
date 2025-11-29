# Min-SNR Loss Reweighting vs Vanilla DDPM


## 1. Question

**Does Min-SNR loss reweighting stabilize training and/or improve sample quality at fixed compute compared to a vanilla ε-prediction DDPM baseline?**

Concretely: for CIFAR-10 (32×32), does using a Min-SNR-style loss weighting improve FID at a fixed number of function evaluations (NFE), and does it produce nicer loss geometry (e.g. lower curvature) during training?

---

## 2. Success metrics

Primary success targets:

- **Sample quality:**  
  - FID@10k (NFE = 50, DDPM sampler) improves by **≥ 7%** vs baseline.
- **Training stability / curvature proxy:**  
  - A curvature proxy (e.g. Hutchinson trace of Hessian approximation, or another agreed-upon metric) decreases by **≥ 10%** vs baseline.

These comparisons are at **matched compute**:
- Same architecture, dataset, β-schedule, total steps, batch size, optimizer, EMA, etc.
- Only the **loss weighting / SNR reweighting** changes.

---

## 3. Baseline setup

Baseline = current “good” DDPM setup from the previous noise-sched work:

- **Data:** CIFAR-10, 32×32, full train set.
- **Model:** `unet_cifar32` (same architecture as the earlier E1/E7 baselines).
- **Diffusion:**
  - ε-prediction DDPM.
  - `beta_schedule: linear`.
  - `num_timesteps: 1000`.
- **Train:**
  - `total_steps: 50_000` (subject to adjustment if needed).
  - `batch_size: 4` (to match previous experiments).
  - `optimizer: adam`, `lr: 1e-4`.
  - Gradient clipping: `grad_clip: 1.0`.
  - Deterministic seed: e.g. `1077`.
- **EMA:**
  - Enabled, `decay: 0.999`.
- **Eval:**
  - Main metric: `val/fid`.
  - Final eval at **NFE = 50**.
  - **10k generated samples** for FID.
  - CIFAR-10 train Inception stats (`stats/cifar10_inception_train.npz`).

This baseline will be wired into `configs/study/MS1_min_snr/E1_baseline_linear.yaml`.

---

## 4. Min-SNR variant(s)

The Min-SNR variant(s) will:

- Keep **all** of the baseline settings fixed (model, data, optimizer, total steps, EMA, sampler, NFE).
- Change **only** the loss weighting to an SNR- or Min-SNR-based scheme.

Initial configs:

- `E1_baseline_linear.yaml` – vanilla ε loss, linear β.
- `E2_minsnr_linear.yaml` – Min-SNR reweighted loss, linear β.

---

## 5. Timebox & cadence

- **Project start:** Wednesday, **December 3, 2025**  
- **Project end:** Monday, **December 29, 2025**

The goals:

1. Some **clear answer** on whether Min-SNR reweighting helps at fixed compute.
2. A **FID vs NFE frontier** comparing baseline vs Min-SNR.
3. **SNR / weighting plots** that help interpret the results.
4. A short written **summary** of findings (2–4 pages in `docs/experiments/`).

---

## 6. Planned experiments 

Exact preregs will live in `docs/experiments/E*_pre_reg.md`, but roughly:

- **E1 – Baseline DDPM / linear β.**  
  Reproduce the existing baseline on this repo.

- **E2 – Min-SNR / linear β.**  
  Same everything, Min-SNR loss reweighting enabled.

- **E3+ – Variants.**  
  - Possibly other β schedules (e.g. cosine).
  - Sensitivity to Min-SNR hyperparameters.
  - NFE-FID frontier sweeps if needed.

Each experiment will be preregistered, run with fixed seeds, and logged to
`runs/` with `results.jsonl` + plots scripts & other misc things under `tools/`.

---

## 7. The Min-SNR project is done when:

1. There is a **baseline vs Min-SNR comparison** at NFE = 50 on CIFAR-10 with:
   - FID@10k for both,
   - Clear statement whether Min-SNR is better, worse, or neutral.
2. There is at least one **NFE ↔ FID plot** comparing curves (if frontier exps were run).
3. There are **SNR / weight vs timestep plots** that make the behavior intuitive.
4. A written **summary** lives in `docs/experiments/` explaining:
   - setup, configs, results, and your interpretation.

Further future work (other schedules, other datasets, etc.) can build on this repo.