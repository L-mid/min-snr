# E24 — Early-t training stress test (Min-SNR)

## What this was testing
Stress test: train only on early timesteps (uniform **t ∈ [0, 50]**) where SNR is high, and check whether **Min-SNR (γ=5)**:
- **Behaviorally** stalls (weights remove learning signal), or
- **Numerically** underflows under AMP (weights become exact 0, gradients vanish).

## Setup (as run)
- CIFAR-10 (subset=4096), UNet CIFAR32 (base_channels=64)
- Cosine β schedule, Adam lr=1e-4, EMA=0.999
- Total steps: 3000
- Seeds: 2222 / 3333 / 4444
- AMP: on
- Variants:
  - **A:** constant loss weighting
  - **B:** Min-SNR weighting, γ=5

## Key results

### Recon metrics @ step 3000 (mean ± sd over 3 seeds)
| Variant     | Recon MSE @ t=10    | Recon MSE @ t=25    | Recon MSE @ t=50    | Recon MSE mean (t=10/25/50)   | Recon PSNR mean (dB)   |
|:------------|:--------------------|:--------------------|:--------------------|:------------------------------|:-----------------------|
| Constant    | 3.126e-04 ± 1.3e-06 | 8.480e-04 ± 5.9e-06 | 1.964e-03 ± 1.4e-05 | 1.041e-03 ± 5.9e-06           | 36.966 ± 0.019         |
| Min-SNR γ=5 | 3.219e-04 ± 1.6e-06 | 8.498e-04 ± 6.4e-06 | 1.934e-03 ± 1.6e-05 | 1.035e-03 ± 6.6e-06           | 36.942 ± 0.023         |

**Takeaway:** Min-SNR did **not** collapse under early-t training. Recon is essentially the same as constant (differences are tiny and within seed-noise for this small run).

### Min-SNR weight stats under early-t sampling (mean over 3 seeds)
|   Step | weight_mean (±sd)   |   weight_min (mean) |   weight_p01 (mean) |   weight_p50 (mean) |   weight_p99 (mean) |   zero_frac (mean) | grad_global_L2 (±sd)   |
|-------:|:--------------------|--------------------:|--------------------:|--------------------:|--------------------:|-------------------:|:-----------------------|
|    250 | 0.0162 ± 0.0006     |           0.0003682 |           0.0006359 |             0.01248 |             0.04076 |                  0 | 0.0025 ± 0.0002        |
|   1000 | 0.0158 ± 0.0006     |           0.0002063 |           0.0002063 |             0.01343 |             0.04065 |                  0 | 0.0033 ± 0.0008        |
|   3000 | 0.0167 ± 0.0013     |           0.0002832 |           0.0004438 |             0.01443 |             0.03999 |                  0 | 0.0036 ± 0.0002        |

**Takeaway:** No AMP underflow in this run:
- `zero_frac` stayed **0.000** at all logged steps.
- weights were small but nonzero (typical mean weight ~0.016–0.017).

### How small are the early-t weights (from the logged Min-SNR curve)?
(γ=5, cosine schedule, K=1000)
- w(t=0)  = 0.000206
- w(t=10) = 0.003608
- w(t=25) = 0.013283
- w(t=50) = 0.041722

This explains why the *reported* `train/loss` and `grad_global_L2` are much smaller under Min-SNR: it’s mostly a scale factor on the learning signal, not a freeze.

## Hypotheses check
- **H1 (behavioral stall):** **Not supported** here. Min-SNR’s recon learning is comparable to constant.
- **H2 (numeric underflow under AMP):** **Not supported** here. `weight_zero_frac` stayed 0; no NaNs/inf/zero-grad detected.


## Interesting:
What an interesting way to end this off.