# E23 eval results (NFE=50, DDIM η=0)

Inputs:
- Baseline arm: e19 checkpoints (constant weighting), 3 seeds
- Min-SNR arm: e20 checkpoints (γ=5), 3 seeds
- Eval: DDIM deterministic, NFE=50, EMA weights, CIFAR-10.
- FID: 50,000 samples; KID: 10,000 pool (subset 1,000 × repeats 10)

## Primary metrics (per seed)

|   seed |   FID baseline |   FID min-SNR |   ΔFID (min-base) |   KID baseline |   KID min-SNR |   ΔKID (min-base) | Both improve?   |
|-------:|---------------:|--------------:|------------------:|---------------:|--------------:|------------------:|:----------------|
|   2222 |         61.053 |        56.331 |            -4.722 |        0.0453  |       0.04426 |          -0.00105 | ✅              |
|   3333 |         62.13  |        54.184 |            -7.946 |        0.04656 |       0.04072 |          -0.00584 | ✅              |
|   4444 |         54.795 |        56.03  |             1.235 |        0.0388  |       0.04153 |           0.00273 | ❌              |

## Across-seed summary (mean ± std)

- **FID@50**: baseline **59.326 ± 3.961**, min‑SNR **55.515 ± 1.162**  
  Δ = **-3.811** (min‑SNR − baseline), i.e. **6.42%** lower FID.

- **KID@50**: baseline **0.04356 ± 0.00416**, min‑SNR **0.04217 ± 0.00185**  
  Δ = **-0.00139** (min‑SNR − baseline), i.e. **3.19%** lower KID.

## Decision rule outcome (per prereg)

- Mean FID improved? **True**
- Mean KID improved? **True**
- Seeds improving on **both** FID & KID: **2/3**
- **Result:** **PASS (confirmed improvement)**

## Recon profile (mean across seeds)

(PSNR higher is better, MSE lower is better)

|   t |   PSNR baseline |   PSNR min-SNR |   ΔPSNR |   MSE baseline |   MSE min-SNR |      ΔMSE |
|----:|----------------:|---------------:|--------:|---------------:|--------------:|----------:|
|  10 |          41.095 |         39.853 |  -1.242 |       0.000311 |      0.000414 |  0.000103 |
| 200 |          26.056 |         26.022 |  -0.034 |       0.009919 |      0.009997 |  7.9e-05  |
| 500 |          20.823 |         20.836 |   0.014 |       0.033099 |      0.032995 | -0.000104 |
| 700 |          18.084 |         18.145 |   0.062 |       0.062192 |      0.06131  | -0.000881 |
| 900 |          14.116 |         14.284 |   0.168 |       0.155048 |      0.149162 | -0.005887 |
| 999 |           5.034 |          5.053 |   0.019 |       1.25503  |      1.24954  | -0.005493 |

Quick read:
- At very low noise (**t=10**), Min‑SNR recon is worse (≈ −1.24 dB PSNR).
- Mid-range (**t=500**) is basically a wash.
- High noise (**t=700–999**) Min‑SNR is slightly better (≈ +0.06 to +0.17 dB PSNR).

## Images/recon at nfe 50:

Litterally almost no notable difference. (shows in FID kinda too, it's down but not much)