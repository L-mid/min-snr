# E1 – Results: Baseline DDPM, linear β, 50k steps (CIFAR-10 32×32)

## Summary

- **Config:** `configs/study/MS1_min_snr/e1/e1_baseline_linear.yaml`
- **Idea:** establish a solid vanilla DDPM baseline for later Min-SNR comparison.
- **Outcome (short):**
  - Final FID@10k (NFE=50): **206.48**
  - Training completed all 50k steps: Yes.
  - Any notable behavior (e.g. early plateaus, instability): None, matched e1 linear baseline.

---

## Quantitative results

### Final metrics

| Metric                         | Value      | Notes                    |
|--------------------------------|------------|--------------------------|
| FID@10k (NFE=50, final eval)   | 206.48   | From `results.jsonl`     |
| Best FID@10k (if different)    | 206.48  | From `results.jsonl`               |
| Final train loss               | 0.003  | From `loss.jsonl`        |


### Training dynamics (qualitative)

- Loss curve shape: Spiky from low batch size but drops steadily.
- Spikes from batch 4 (as expected).
- Intermittent FID milestones: (maybe) increase over time but within noise band, all in the same ballpark.

---

## Artifacts

- **Plot:**
    - ![alt text](../../assets/e1/e1_plots/loss_fid_overlay.png)

- **Sample grids:**
  [alt text](../../assets/e1/min-snr-e1/runs/min_snr/min_snr__unet_cifar32__cifar10__adam__lr1e-04__ema1__seed=1077/eval/step_050000/grid/grid.png)
  
  - Quick impression of image quality: not tv static, saturated blobs consistent with baseline.

---

## Interpretation

- **Reasonable baseline?**
  - FIDs extremely similar. 
  - No surprises: deterministic replica

Might want to consider some overall later improvements (model underfits sevearly is my guess), but for testing purposes this is fine and expected.

