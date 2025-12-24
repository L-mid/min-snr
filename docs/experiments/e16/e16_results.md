# Results — E16 seed sensitivity (with E14 + E15 comparators)

This file summarizes the prereg endpoints for **E16** and compares against:
- **E14**: min-SNR (γ=5), linear β schedule, **NFE=50**
- **E15**: constant weighting (baseline), linear β schedule, **NFE=25**

All three use the same seeds and evaluation cadence.

---

## Experiment IDs

- **E14** = min-snr, NFE 50
- **E15** = constant weighting, NFE 25 (baseline)
- **E16** = min-snr, NFE 25

**Seeds:** {1, 11, 1077}  
**Milestones:** steps 2,500 / 5,000 / 7,500 / 10,000  
**Primary endpoints:** FID@10k, KID@10k, plus seed sensitivity (sd/range).

---

## Primary endpoints @ step 10,000

Mean ± sd over 3 seeds, with (min–max) range.

| exp | setting | FID @10k | KID @10k |
|---|---|---:|---:|
| **E14** | min-snr, **NFE=50** | **368.695 ± 0.016** (368.682–368.713) | **0.04991 ± 0.00255** (0.04830–0.05285) |
| **E15** | constant, **NFE=25** | **368.756 ± 0.008** (368.750–368.765) | **0.09880 ± 0.00564** (0.09414–0.10507) |
| **E16** | min-snr, **NFE=25** | **369.203 ± 0.007** (369.198–369.211) | **0.06153 ± 0.00317** (0.05898–0.06508) |

### Seed sensitivity (E16)

At step 10k:
- **FID sd = 0.00745**, range = 0.01323
- **KID sd = 0.00317**, range = 0.00610

Interpretation: for E16, seed variability is *very small*; changes much larger than ~0.01 FID are unlikely to be seed noise in this setting.

---

## Milestone means (trend across training)

### FID means (lower is better)
- **E14 (NFE50, min-snr):** 367.712 → 368.582 → 368.664 → 368.695
- **E15 (NFE25, constant):** 368.172 → 368.643 → 368.729 → 368.756
- **E16 (NFE25, min-snr):** 368.678 → 369.139 → 369.184 → 369.203

### KID means (lower is better)
- **E14 (NFE50, min-snr):** 0.1717 → 0.1067 → 0.0755 → 0.0499
- **E15 (NFE25, constant):** 0.1837 → 0.1276 → 0.1069 → 0.0988
- **E16 (NFE25, min-snr):** 0.1738 → 0.1149 → 0.0895 → 0.0615

**Pattern:** across all three, FID worsens with more training steps, while KID improves.

---

## Comparator deltas @ step 10,000

Define Δ = (row A − row B). Positive means A is worse.

### NFE effect within min-snr (E16 vs E14)
- **ΔFID = +0.507** (E16 worse)
- **ΔKID = +0.0116** (E16 worse)

Interpretation: **NFE=25 is not metric-equivalent to NFE=50** here; the shift is much larger than the E16 seed sd.

### Weighting effect at NFE=25 (E16 vs E15)
- **ΔFID = +0.447** (min-snr worse)
- **ΔKID = −0.0373** (min-snr better)

Interpretation: at NFE=25, min-snr strongly improves **KID**, but hurts **FID** in this setup.

---

## Secondary: wall-clock + eval share

Mean over 3 seeds.

| exp | setting | total time | sec/step | eval time | eval share |
|---|---|---:|---:|---:|---:|
| E14 | min-snr, NFE50 | 53.34 min | 0.320 s/step | 26.77 min | 0.502 |
| E15 | constant, NFE25 | 40.96 min | 0.246 s/step | 15.04 min | 0.367 |
| E16 | min-snr, NFE25 | 28.28 min | 0.170 s/step | 9.74 min | 0.345 |

Interpretation: E16 achieves the intended speed win:
- ~**1.9× faster total** than E14
- ~**2.7× faster eval time** than E14

---

### Samples:

#### seed 1 (e16, min-snr, nfe 25)
![alt text](../../assets/e15/min-snr-e15/runs/min_snr/min_snr__unet_cifar32__cifar10__adam__lr1e-04__ema1__seed=1/eval/step_010000/grid/grid.png)


#### seed 1 (e14, min-snr, nfe 50)
![alt text](../../assets/e14/min-snr-e14/runs/min_snr/min_snr__unet_cifar32__cifar10__adam__lr1e-04__ema1__seed=1/eval/step_010000/grid/grid.png)


## So:

- **E16 seed variability is tiny** (stable noise floor).
- **NFE=25 is much cheaper** in wall-clock/eval time than NFE=50.
- **NFE=25 metrics are systematically shifted vs NFE=50**.
- At NFE=25, **min-snr improves KID but worsens FID** vs constant weighting (E16 vs E15), in this run family (consistent with prior results).
