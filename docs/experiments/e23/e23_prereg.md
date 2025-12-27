
# E23 — Confirmation-grade eval @ NFE=50 (baseline vs Min-SNR γ=5)

**Preregistered:** 2025-12-27 (Europe/London)  
**Study:** min_snr (CIFAR-10, UNet-CIFAR32)  
**Type:** Eval-only re-measurement (no retraining)  
**Related runs:** e19 (constant baseline), e20 (Min-SNR γ=5)

---

## 0) Motivation

We already trained hold-out runs for:
- **Baseline:** cosine β schedule + constant weighting (e19)
- **Min-SNR:** cosine β schedule + Min-SNR weighting (γ=5) (e20)

Those runs logged metrics at **DDIM NFE=20**.  
E23 exists to produce a reporting-grade, apples-to-apples evaluation at **DDIM NFE=50**, with lower-variance KID and higher-sample FID, so we can confidently explain the result vs a typical baseline.

---

## 1) Primary question

When evaluated with DDIM (η=0) at NFE=50, does Min-SNR γ=5 outperform constant weighting on CIFAR-10 under the same eval protocol?

---

## 2) Conditions (arms)

**Arm A (Baseline):**
- e19 checkpoint set (3 seeds): constant loss weighting
- cosine β schedule
- EMA enabled (evaluate EMA weights)

**Arm B (Min-SNR):**
- e20 checkpoint set (3 seeds): Min-SNR loss weighting with γ=5
- cosine β schedule
- EMA enabled (evaluate EMA weights)

**Experimental unit:** one final checkpoint (`ckpts/last.pt`) for a given seed.  
**Seeds:** `[2222, 3333, 4444]` (exactly).

---

## 3) Fixed settings (must match across arms)

- Dataset: CIFAR-10
- Model: `unet_cifar32`, `base_channels=64`
- Training steps: 10k (already done; eval-only here)
- Beta schedule: cosine
- Sampler: **DDIM deterministic (η=0)**
- Evaluation sampler NFE: **50**
- Use EMA weights: **ON**
- AMP: **ON**
- Optional safety: `clip_x0: ON` (keep consistent for both arms)

---

## 4) Metrics + evaluation protocol (confirmation-grade)

### 4.1 Primary metrics
**FID@NFE=50**
- Generated samples: **50,000**
- Stats file: `stats/cifar10_inception_train.npz`
- Report per-seed FID and mean±std across seeds.

**KID@NFE=50**
- Generated samples pool: **10,000**
- `subset_size`: **1,000**
- `repeats`: **10**
- Real features: CIFAR-10 train, fixed seed (cache allowed)
- Report per-seed KID mean and mean±std across seeds.

### 4.2 Diagnostic / explainer metrics
**Recon profile (fixed t)**
- t values: `[10, 200, 500, 700, 900, 999]`
- `n_batches`: 8
- metrics: MSE, PSNR
- Purpose: explain *where* denoising quality shifts along t.

**Grid samples**
- `n_samples`: 36
- sampler: DDIM NFE=50
- Purpose: qualitative sanity + “typical vs improved” visuals.

---

## 5) Decision rules (success / fail / inconclusive)

E23 is a confirmation experiment (not a new sweep).

**PASS (confirmed improvement):**
- Mean FID (Min-SNR) < mean FID (Baseline), AND
- Mean KID (Min-SNR) < mean KID (Baseline), AND
- At least **2/3 seeds** improve on FID **and** KID.

**FAIL (not confirmed):**
- Either metric gets worse on mean, OR
- Improvements occur in <2/3 seeds for either metric.

**INCONCLUSIVE:**
- Mixed results (e.g., FID improves but KID worsens) or highly overlapping seed variance.
- If inconclusive, we still ship e23 outputs and note the ambiguity.

No p-hacking: no changing seeds, no dropping bad seeds.

---

## 6) Implementation plan (CLI, GPU)

We run a single eval-only script on GPU for each arm, pointing at the **final checkpoints**.

**Script:** `tools/e23/eval_ckpts_nfe.py`  
(Deterministic DDIM, evenly-spaced integer timesteps; same codepath for both arms.)

### 6.1 Commands

**Arm A — Baseline (e19):**
```bash
python tools/e23/eval_ckpts_nfe.py \
  --label e23_e19_baseline \
  --cfg <PATH_TO_E19_YAML> \
  --ckpt <PATH_TO_E19_SEED2222_LAST.PT> \
  --ckpt <PATH_TO_E19_SEED3333_LAST.PT> \
  --ckpt <PATH_TO_E19_SEED4444_LAST.PT> \
  --out runs/e23/e19_baseline_nfe50_confirm \
  --nfe 50 \
  --fid_stats stats/cifar10_inception_train.npz \
  --fid_n_samples 50000 \
  --kid_n_samples 10000 --kid_subset 1000 --kid_repeats 10 \
  --kid_cache_real runs/e23/_cache/kid_real_feats_train_seed123.npz \
  --grid_n 36 \
  --recon_batches 8 --recon_t 10 200 500 700 900 999 \
  --use_ema --amp --clip_x0
