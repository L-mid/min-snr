# Results: E15 — Seed sensitivity (constant/baseline, linear) vs E13

## TL;DR
- E15 confirms **seed variance is tiny** for FID (and modest for KID), especially by 10k.

---

## E15 setup (as-run)
- Seeds: {1, 11, 1077}
- CIFAR-10 full, bs=64, workers=2
- UNet CIFAR32, base_channels=64, EMA=0.999
- beta_schedule=linear
- loss weighting = constant
- Train steps = 10,000
- Eval milestones at steps: 2.5k / 5k / 7.5k / 10k
  - KID: DDIM, NFE=25, n=5000, repeats=20
  - FID milestone: DDPM, n=5000 

---

## E15: Seed variability (primary endpoint)
(3 seeds; std = sample std across seeds)

### FID across seeds (very tight)
| step | FID mean | FID std | range (max-min) |
|---:|---:|---:|---:|
| 2,500 | 368.172 | 0.076 | 0.150 |
| 5,000 | 368.643 | 0.018 | 0.037 |
| 7,500 | 368.729 | 0.0099 | 0.019 |
| 10,000 | 368.756 | 0.0078 | 0.015 |

### KID across seeds (more variance than FID, still small)
| step | KID mean | KID std | range (max-min) |
|---:|---:|---:|---:|
| 2,500 | 0.1837 | 0.0191 | 0.0367 |
| 5,000 | 0.1276 | 0.0044 | 0.0078 |
| 7,500 | 0.1069 | 0.0042 | 0.0083 |
| 10,000 | 0.0988 | 0.0056 | 0.0109 |

**Conclusion:** By 10k, seed noise is negligible for FID and modest for KID. Using 1 seed for iteration is  (as assumed before) good enough, 3 seeds is confirmatory.

---

## E15: Timing (as-run)
Per milestone (mean across seeds):
- FID time ≈ 109 s
- KID time ≈ 106 s (first milestone had one slow outlier; later milestones stable)
- FID+KID per milestone ≈ 215 s (~3.6 min)

Per run totals (mean across seeds):
- Total runtime: 40:58
- Total eval: 15:02
- Eval fraction: 36.7%

---

## Compare: E13 vs E15 (same seeds)
**Caveat:** E13 was run in a different environment than E15 (kaggle vs colab GPUs), so wall-clock speedups are not purely NFE effects. Pinned Torch + implementations remain the same.

### Wall-clock (mean across seeds)
| Experiment | total runtime | eval total | eval fraction |
|---|---:|---:|---:|
| E13 | 51:54 | 26:06 | 50.3% |
| E15 | 40:58 | 15:02 | 36.7% |

### Per-milestone eval speedup (E13 / E15)
- ~1.76× faster (FID+KID combined) across milestones.

### Metric shifts (E15 − E13, mean across seeds)
| step | ΔFID | ΔKID |
|---:|---:|---:|
| 2,500 | +0.625 | +0.0013 |
| 5,000 | +0.230 | +0.0078 |
| 7,500 | +0.189 | +0.0165 |
| 10,000 | +0.159 | +0.0249 |


### Samples (seed 1, ddim):


#### Seed 1, nfe 25 (e15):

![alt text](../../assets/e13/e13_media/1_grid.png)

#### Seed 1, nfe 50 (e13):

![alt text](../../assets/e15/min-snr-e15/runs/min_snr/min_snr__unet_cifar32__cifar10__adam__lr1e-04__ema1__seed=1/eval/step_010000/grid/grid.png)


Interpretation:
- KID shift is expected given E15 uses KID NFE=25 vs E13 KID NFE=50.
- FID shift is small.
- linear baseline is stable under small (25) nfe under multiple seeds.


