# [ID] – [e1_baseline_linear]

## Summary

- **Question:**  
  Establish a vanilla ε-prediction DDPM baseline on CIFAR-10 (32×32) with a linear β schedule and NFE=50, to compare directly against Min-SNR loss reweighting later.

- **Hypothesis / expectation:**  
  Should be deterministic, based on prev results this baseline should be undertrained with a garbage fid, but functional to compare against. 

- **Primary comparison:**  
  E2 (Min-SNR linear, 50k) will be compared directly to this E1 baseline at matched compute and NFE.

---

## Configuration

- **Config path:**  
  `configs/study/MS1_min_snr/e1/e1_baseline_linear.yaml`

- **Key knobs:**
  - Model: `unet_cifar32`
  - Dataset: CIFAR-10, 32×32, full train set
  - Beta schedule: `linear`
  - Diffusion steps: `num_timesteps: 1000` 
  - Loss: standard ε-prediction DDPM MSE loss
  - Sampler: `ddpm` at eval, `nfe: 50`
  - Total steps: `50_000`
  - Batch size: `4`
  - Optimizer: `adam`, `lr: 1e-4`
  - EMA: enabled, `decay: 0.999`
  - AMP: `true`

- **Seed(s):**
  - Train seed: `1077`

---

## Metrics & success criteria

- **Primary metric(s):**
  - FID@10k (NFE = 50) using CIFAR-10 train Inception stats
  - Training loss curve (for sanity)

- **Success criteria:**
  - The run completes 50k steps without divergence.
  - Final FID@10k is in a reasonable range for this model & setup (exact value not pre-specified, but should be consistent with similar runs in noise-sched).
  - This experiment becomes the “baseline row” in the Min-SNR vs vanilla comparison table.


---

## DOD

Commands to run:
   ```bash
   # example
   python -m ablation_harness.cli run \
     --config configs/study/MS1_min_snr/configs/study/MS1_min_snr/e1/e1_baseline_linear.yaml \
     --out_dir /content/drive/MyDrive/min-snr-e1/runs