# [ID] – [Short experiment name]

## Summary

- **Question:**  
  What specific question does this experiment answer?

- **Hypothesis / expectation:**  
  What do you expect to see, and why?

- **Primary comparison:**  
  vs which baseline / prior experiment?

---

## Configuration

- **Config path:**  
  `configs/study/MS1_min_snr/[...]`

- **Key knobs:**
  - Model: `[...]`
  - Beta schedule: `[...]`
  - Loss / weighting: `[...]`
  - Sampler + NFE: `[...]`
  - Total steps: `[...]`
  - Batch size: `[...]`
  - EMA: `[...]`

- **Seed(s):**
  - Train seed: `[...]`

---

## Metrics & success criteria

- **Primary metric(s):**
  - e.g. FID@10k (NFE=50), train loss, curvature proxy, etc.

- **Success criteria:**
  - e.g. “FID@10k improves by ≥ X% vs E1 baseline at same NFE”
  - e.g. “Curvature proxy decreases by ≥ Y% vs baseline”

---

## Command(s) to run:
   ```bash
   # example
   python -m ablation_harness.cli run \
     --config configs/study/MS1_min_snr/[...].yaml \
     --out_dir runs/[...]