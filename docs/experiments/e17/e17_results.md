# Results: E17 — Seed sensitivity (baseline, cosine β, NFE=25)

**Important note (qualitative grids):** some **failed/corrupted grids** due to a sampling issue. For qualitative judgment, using resampled grids from seed=1’s checkpoint, not the grids inside the original run artifact.

## Prereg (intent)
- Question: how much do random seeds change results under the baseline (cosine schedule)?
- Seeds: `[1077, 11, 1]`
- Train: 10k steps, batch 64, EMA=0.999, AMP on, grad_clip=1.0
- Eval key: KID/FID every 2500 steps (DDIM, NFE=25), plus grids/recon for sanity.

## Metrics (per-seed @ milestone steps)
KID/FID are logged at steps: **2500 / 5000 / 7500 / 10000**.

| seed | FID@2500 | KID@2500 | FID@10000 | KID@10000 |
|---:|---:|---:|---:|---:|
| 1    | 299.87 | 0.3468 | 327.41 | 0.2972 |
| 11   | 277.36 | 0.3038 | 329.84 | 0.2994 |
| 1077 | 289.61 | 0.3325 | 327.76 | 0.2993 |

## Seed spread (mean ± std across 3 seeds)
| step | FID (mean ± std) | KID (mean ± std) |
|---:|---:|---:|
| 2500  | 288.95 ± 9.20 | 0.3277 ± 0.0179 |
| 5000  | 313.84 ± 2.33 | 0.3146 ± 0.0015 |
| 7500  | 323.63 ± 0.59 | 0.3058 ± 0.0001 |
| 10000 | 328.34 ± 1.07 | 0.2986 ± 0.0010 |

**Ranges (max−min across seeds):**
- Step 2500: FID range **22.51**, KID range **0.0430**
- Step 10000: FID range **2.43**, KID range **0.00220**

## Trend
- **FID worsens with training after step 2500** for all seeds (best FID occurs at 2500 in each run).
- **KID generally improves** as training continues (though seed=11 has a small bump at 5000 before improving).


## Runtime (per seed)
| seed | wall (min) | eval (min) | eval % |
|---:|---:|---:|---:|
| 1    | 27.88 | 9.66 | 34.63% |
| 11   | 27.90 | 9.66 | 34.61% |
| 1077 | 28.26 | 9.88 | 34.96% |

Eval time is dominated by **KID (~4.5–4.8 min)** and **FID (~4.7 min)**.

## Grids:
- Run grids from seed 1 (impacted by the known cosine sampling issue):

  - `.../eval/step_010000/grid/grid.png` (and similarly for other steps/seeds)

![alt text](../../assets/e17/min-snr-e17-runs/runs/min_snr/min_snr__unet_cifar32__cifar10__adam__lr1e-04__ema1__seed=1/eval/step_010000/grid/grid.png)


## Acutal (resampled from seed 1 ckpt:)

![alt text](../../assets/e17/seed1.png)


## So:

About the same as linear so far (might change with direct comparasion, but I'll leave it at that for now).

