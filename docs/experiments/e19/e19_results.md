# Results: e19 (hold-out baseline) + e17 (baseline comparison)

This file summarizes this e19 and e17 thing.
---

## Common training setup (both e17 and e19)

- Dataset: CIFAR-10 (32×32), batch_size=64
- Model: `unet_cifar32`, base_channels=64
- Optim: Adam lr=1e-4, grad_clip=1.0, AMP=True
- Diffusion β schedule: cosine
- EMA: enabled, decay=0.999
- Loss weighting: constant
- Total steps: 10,000

---

# E19 — Hold-out baseline (constant weighting), seeds 2222/3333/4444

**Seeds:** 2222, 3333, 4444  

## Eval settings (from cfg)
- Grid: sampler=DDIM, NFE=20, every=1000, n_samples=36
- Recon: every=2000, `t_values=[0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]`
- KID: sampler=DDIM, NFE=20, every=2500, n_samples=5000, repeats=5
- FID milestone: sampler=DDIM, NFE=20, every=2500, n_samples=5000, gate=0.0 (always run)

## FID @ milestones
| seed | 2.5k | 5k | 7.5k | 10k |
|---:|---:|---:|---:|---:|
| 2222 | 78.51 | 76.84 | 86.06 | 83.53 |
| 3333 | 72.55 | 82.21 | 91.48 | 87.21 |
| 4444 | 80.89 | 75.70 | 77.55 | 74.61 |

## KID @ milestones
| seed | 2.5k | 5k | 7.5k | 10k |
|---:|---:|---:|---:|---:|
| 2222 | 0.07196 | 0.05957 | 0.07247 | 0.07084 |
| 3333 | 0.06459 | 0.06623 | 0.08004 | 0.07611 |
| 4444 | 0.07764 | 0.05691 | 0.06131 | 0.06207 |

## Across-seed summary (n=3; sample std)

**FID**
| checkpoint | mean | std | min | max |
|---:|---:|---:|---:|---:|
| 2500 | 77.32 | 4.29 | 72.55 | 80.89 |
| 5000 | 78.25 | 3.48 | 75.70 | 82.21 |
| 7500 | 85.03 | 7.02 | 77.55 | 91.48 |
| 10000 | 81.79 | 6.48 | 74.61 | 87.21 |

**KID**
| checkpoint | mean | std | min | max |
|---:|---:|---:|---:|---:|
| 2500 | 0.07140 | 0.00654 | 0.06459 | 0.07764 |
| 5000 | 0.06090 | 0.00480 | 0.05691 | 0.06623 |
| 7500 | 0.07127 | 0.00942 | 0.06131 | 0.08004 |
| 10000 | 0.06967 | 0.00709 | 0.06207 | 0.07611 |

## Best checkpoint per seed (lowest FID across the 4 milestones)
| seed | best_fid | best_step | kid_at_best |
|---:|---:|---:|---:|
| 2222 | 76.84 | 5000 | 0.05957 |
| 3333 | 72.55 | 2500 | 0.06459 |
| 4444 | 74.61 | 10000 | 0.06207 |

## Runtime
- **Runtime:** 26m30s ± 12.6s  
- **sec/step:** 0.159  
- **eval share:** 31.2%

| seed | run_time | sec/step | eval_frac | eval_s_total | kid_s | fid_s | grid_s | recon_s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2222 | 26m44s | 0.160 | 31.5% | 505.3s | 236.0s | 244.3s | 3.6s | 17.6s |
| 3333 | 26m23s | 0.158 | 31.0% | 491.0s | 228.3s | 238.4s | 3.4s | 17.6s |
| 4444 | 26m22s | 0.158 | 31.0% | 490.0s | 230.1s | 241.6s | 3.4s | 17.3s |


# E17 — Baseline (constant weighting), seeds 1/11/1077

**Seeds:** 1, 11, 1077  

## Eval settings (from cfg)
- Grid: sampler=DDIM, NFE=25, every=1000, n_samples=36
- Recon: every=2000, `t_values=[0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]`
- KID: sampler=DDIM, NFE=25, every=2500, n_samples=5000, repeats=20
- FID milestone: sampler=DDPM, NFE=25, every=2500, n_samples=5000, gate=0.0 (always run)

## FID @ milestones
| seed | 2.5k | 5k | 7.5k | 10k |
|---:|---:|---:|---:|---:|
| 1 | 299.87 | 316.32 | 323.77 | 327.41 |
| 11 | 277.36 | 314.49 | 324.27 | 329.84 |
| 1077 | 289.61 | 310.73 | 322.84 | 327.76 |

## KID @ milestones
| seed | 2.5k | 5k | 7.5k | 10k |
|---:|---:|---:|---:|---:|
| 1 | 0.34682 | 0.31668 | 0.30587 | 0.29716 |
| 11 | 0.30380 | 0.31303 | 0.30573 | 0.29936 |
| 1077 | 0.33252 | 0.31412 | 0.30566 | 0.29927 |

## Across-seed summary (n=3; sample std)

**FID**
| checkpoint | mean | std | min | max |
|---:|---:|---:|---:|---:|
| 2500 | 288.95 | 11.27 | 277.36 | 299.87 |
| 5000 | 313.84 | 2.85 | 310.73 | 316.32 |
| 7500 | 323.63 | 0.72 | 322.84 | 324.27 |
| 10000 | 328.34 | 1.31 | 327.41 | 329.84 |

**KID**
| checkpoint | mean | std | min | max |
|---:|---:|---:|---:|---:|
| 2500 | 0.32771 | 0.02239 | 0.30380 | 0.34682 |
| 5000 | 0.31461 | 0.00190 | 0.31303 | 0.31668 |
| 7500 | 0.30575 | 0.00011 | 0.30566 | 0.30587 |
| 10000 | 0.29860 | 0.00119 | 0.29716 | 0.29936 |

## Best checkpoint per seed (lowest FID across the 4 milestones)
| seed | best_fid | best_step | kid_at_best |
|---:|---:|---:|---:|
| 1 | 299.87 | 2500 | 0.34682 |
| 11 | 277.36 | 2500 | 0.30380 |
| 1077 | 289.61 | 2500 | 0.33252 |

## Runtime
- **Runtime:** 27m55s ± 8.8s  
- **sec/step:** 0.167  
- **eval share:** 34.9%

| seed | run_time | sec/step | eval_frac | eval_s_total | kid_s | fid_s | grid_s | recon_s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27m48s | 0.167 | 34.6% | 578.6s | 275.1s | 282.4s | 4.3s | 17.3s |
| 11 | 28m02s | 0.168 | 35.4% | 593.9s | 283.7s | 287.0s | 4.4s | 17.5s |
| 1077 | 28m00s | 0.168 | 34.8% | 582.0s | 270.6s | 283.1s | 4.4s | 17.3s |


# E19 vs E17: notes

- Both runs use the same training setup (constant weighting, cosine β, EMA, 10k steps, etc.).
- Eval differs materially:
  - e19: KID/FID milestone use DDIM NFE=20 and KID repeats=5.
  - e17: FID milestone uses DDPM NFE=25, KID uses DDIM NFE=25 and repeats=20.
- Runtime is lower in e19 largely because eval is lighter:
  - e19 mean sec/step ≈ 0.159
  - e17 mean sec/step ≈ 0.167


Most notably: Exceedingly high FIDs in prev experiments seem diagnosable to DDPM sampling used with a low nfe (which also creates noticably worse samples), instead of DDIM.

### e19, seed 1:

![alt text](../../assets/e19/runs/min_snr/min_snr__unet_cifar32__cifar10__adam__lr1e-04__ema1__seed=2222/eval/step_010000/grid/grid.png)


Cifarish. 
