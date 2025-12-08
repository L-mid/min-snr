# [W1-E6] – Min-SNR Hutchinson trace probe (linear β, 10k, 50 NFE)

## Summary

- **Question:**  
  How does Min-SNR loss reweighting change the curvature trajectory (Hutchinson trace of the loss) during training compared to a vanilla constant-weight ε-MSE baseline at matched compute (10k steps, linear β, same model), and does that line up with any shift in FID / training loss behaviour?

- **Hypothesis / expectation:**  
  - Min-SNR will down-weight low-SNR tail timesteps and put more mass on the more informative mid-SNR region.  
  - As a result, the Hutchinson trace (as a rough curvature proxy) should:
    - Be lower on average after the early burn-in (e.g. 2k–10k) vs W1-E5, and/or  
    - Show fewer large spikes / less volatility at late steps where constant weighting is still chasing noisy gradients.  
  - FID is expected to still be undertrained-bad!, but:
    - Final FID@10k for W1-E6 should be no worse than W1-E5, and may be slightly better if Min-SNR makes optimization a bit friendlier. No change acceptable.

- **Primary comparison:**  
  - **Primary:** W1-E6 vs W1-E5 (same everything, only `loss.weighting` changes).  
  - **Secondary context:** compare curvature + FID patterns to:
    - W1-E1 (vanilla linear baseline, no curvature logging), and  
    - W1-E2 (Min-SNR 50k run) to see if the curvature story at 10k is consistent with the longer Min-SNR run.

---

## Configuration

- **Config path:**  
  `configs/study/MS1_min_snr/e6/e6_minsnr_linear_hutch_trace_10k_50nfe.yaml`

- **Key knobs:**
  - Model: `unet_cifar32` (same as E1/E2/E5)
  - Beta schedule: `"linear"`
  - Loss / weighting: `loss.weighting: "minsnr"` (same Min-SNR settings as E2)
  - Sampler + NFE:
    - Milestone FID & final: `ddpm`, `nfe: 50`
    - Grid samples: `ddim`, `nfe: 20`
  - Total steps: `10000`
  - Batch size: `4`
  - EMA: `enabled: true`, `decay: 0.999`

- **Seed(s):**
  - Train seed: `1077` (match W1-E5 for clean comparison)

---

## Metrics & success criteria

- **Primary metric(s):**
  - `curvature/hutch_trace_mean` vs `global_step`
  - `curvature/hutch_trace_std` vs `global_step`
  - Training loss metrics:
    - `train/loss` 
  - FID:
    - `val/fid_milestone/fid` at {2k, 4k, 6k, 8k, 10k}
    - `eval/final/fid` at 10k, NFE=50

- **Success criteria:**

  **Curvature assessment:**
  - After burn-in (from **step 2k onward**), at least one of:
    - The mean Hutchinson trace for W1-E6 is lower than W1-E5, e.g.  
      \- Average `curvature/hutch_trace_mean` over steps 6k–10k is ≥ 15% lower than W1-E5, or  
    - W1-E6 shows reduced spikiness: fewer large curvature spikes (e.g. fewer outlier points > 1.5× the median W1-E5 trace in the same region).
  - Qualitatively, plots should make it clear that Min-SNR has changed the curvature trajectory, not just produced random noise.

  **FID / optimization sanity (secondary):**
  - Final FID@10k (NFE=50, ddpm) for W1-E6 is:
    - Not worse than +10% relative to W1-E5 final FID, and  
    - Ideally improved by ≥ 5–10%, but “no worse while curvature changes” is already interesting.

  If curvature clearly shifts in the expected direction while FID stays within the ±10% band of W1-E5, the experiment is considered a useful probe even if absolute sample quality is still garbage.

---

## Command(s) to run:

```bash
python -m ablation_harness.cli run \
  --config configs/study/MS1_min_snr/e6/e6_minsnr_linear_hutch_trace_10k_50nfe.yaml \
  --out_dir runs/W1_E6_minsnr_linear_hutch_trace_10k_50nfe      # example out_dir
