# Results — e22 (holdout confirm, Min-SNR γ=5, cosine)

## What was run (from prereg)
- CIFAR-10, UNet CIFAR32 (base_channels=64), batch size 64  
- Adam lr=1e-4, 10k steps, AMP, grad_clip=1.0, EMA=0.999  
- Diffusion β schedule: cosine  
- Loss weighting: Min-SNR, γ=5  
- Holdout seeds: 2222 / 3333 / 4444  
- Intended eval: NFE=35 (DDIM) for grid/KID/FID milestone

**Run command:**
`python -m ablation_harness.cli run --config configs/study/MS1_min_snr/e22/e22_best_min-snr_nfe_35.yaml`

## Primary metric — FID milestone (↓ better)

| seed   |   2500 |   5000 |   7500 |   10000 |
|:-------|-------:|-------:|-------:|--------:|
| 2222   |  88.25 |  75.08 |  72.43 |   61.8  |
| 3333   |  86.68 |  75    |  70.07 |   61.58 |
| 4444   |  89.2  |  70.74 |  72.28 |   66.38 |
| mean   |  88.04 |  73.61 |  71.59 |   63.25 |
| std    |   1.28 |   2.48 |   1.32 |    2.71 |

**10k-step holdout summary (n=3 seeds):** FID = **63.25 ± 2.71**  
Per-seed @10k: 2222=61.80, 3333=61.58, 4444=66.38

## Secondary — KID (↓ better)

| seed   |    2500 |    5000 |    7500 |   10000 |
|:-------|--------:|--------:|--------:|--------:|
| 2222   | 0.0874  | 0.05494 | 0.05414 | 0.04592 |
| 3333   | 0.08715 | 0.05712 | 0.05262 | 0.04537 |
| 4444   | 0.09246 | 0.05206 | 0.05329 | 0.04907 |
| mean   | 0.089   | 0.05471 | 0.05335 | 0.04679 |
| std    | 0.003   | 0.00254 | 0.00076 | 0.002   |

**10k-step holdout summary (n=3 seeds):** KID = **0.04679 ± 0.00200**

## Recon diagnostics (fixed t list) @ 10k

|   seed |   mse_fixed_mean |   psnr_fixed_mean |
|-------:|-----------------:|------------------:|
|   2222 |         0.251489 |           20.6948 |
|   3333 |         0.248442 |           20.6727 |
|   4444 |         0.252504 |           20.6944 |

Mean across seeds @10k:
- recon MSE (fixed mean): **0.250812**
- recon PSNR (fixed mean): **20.687**

## Eval wall-clock (seconds spent inside eval at each FID/KID milestone)

| seed   |   2500 |   5000 |   7500 |   10000 |
|:-------|-------:|-------:|-------:|--------:|
| 2222   |  201   |  187.4 |  185.9 |   189.6 |
| 3333   |  185.8 |  186.2 |  186.3 |   190.3 |
| 4444   |  186.1 |  186.4 |  185.6 |   189.8 |
| mean   |  191   |  186.7 |  185.9 |   189.9 |
| std    |    8.7 |    0.7 |    0.3 |     0.4 |

Notes:
- step 2500 is noticeably slower (one-time overheads).
- later milestones stabilize around **~186–190s** total per milestone (≈95s FID + ≈91s KID).

## End-to-end runtime per seed (W&B _runtime)

| seed | runtime (min) |
|---:|---:|
| 2222 | 31.86 |
| 3333 | 31.34 |
| 4444 | 31.36 |
| **mean ± std** | **31.52 ± 0.30** |


## Interesting result:
Fid + kid (but especally FID) go down more consistently over time with the higher NFE than NFE 20 results.
