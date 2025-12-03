# E1 – Results: Baseline DDPM, linear β, 50k steps (CIFAR-10 32×32)

## 1. Summary

- **Config:** `configs/study/MS1_min_snr/E1_baseline_linear.yaml`
- **Out dir:** `runs/E1_baseline`
- **Question:** establish a solid vanilla DDPM baseline for later Min-SNR comparison.
- **Outcome (short):**
  - Final FID@10k (NFE=50): **TODO_FID**  
  - Training completed all 50k steps: **TODO_YES_NO**
  - Any notable behavior (e.g. early plateaus, instability): **TODO_NOTES**

---

## 2. Quantitative results

### 2.1 Final metrics

| Metric                         | Value      | Notes                    |
|--------------------------------|------------|--------------------------|
| FID@10k (NFE=50, final eval)   | TODO_FID   | From `results.jsonl`     |
| Best FID@10k (if different)    | TODO_BEST  | Optional                 |
| Final train loss               | TODO_LOSS  | From `loss.jsonl`        |

Add more rows if you care (e.g. KID, curvature proxy later).

### 2.2 Training dynamics (qualitative)

- Loss curve shape: **TODO_DESCRIPTION** (e.g., “steady decrease, mild noise”).
- Any obvious signs of divergence / weird spikes: **TODO_DESCRIPTION**.
- Intermittent FID milestones: **TODO_BRIEF_NOTES**.

---

## 3. Artifacts

- **Logs:**
  - `runs/E1_baseline/.../loss.jsonl`
  - `runs/E1_baseline/.../results.jsonl`

- **Plots:**
  - TODO: after you add plot scripts, link:
    - `docs/assets/E1/loss_curve.png` (planned)
    - `docs/assets/E1/fid_vs_step.png` (planned)

- **Sample grids:**
  - Path(s): `runs/E1_baseline/.../grid_*.png` (if generated)
  - Quick impression of image quality: **TODO_DESCRIPTION**

---

## 4. Interpretation

- **Is this a reasonable baseline?**
  - Does the FID sit in the ballpark of your previous noise-sched E7 baseline?
  - Any surprises vs expectations?

- **Anything to fix before Min-SNR runs?**
  - Config issues?
  - Logging / plotting gaps?
  - Data problems?

Write a short paragraph or bullet list once you’ve looked at the numbers.

---

## 5. Next steps

- If baseline looks sane:
  - Lock E1 as the baseline row.
  - Proceed to designing & wiring Min-SNR loss weighting (E2).

- If baseline is off:
  - Diagnose (data, model, β schedule, stats file, etc.).
  - Re-run E1 after fixes, and update this file.