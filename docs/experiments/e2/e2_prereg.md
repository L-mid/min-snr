# [ID] – E2 Min-SNR vs baseline, linear β, 50k steps

## Question

On a small, underpowered CIFAR-10 DDPM, does Min-SNR loss reweighting improve sample quality or training behaviour compared to a vanilla ε-MSE loss at matched compute?

Concretely: with **all hyperparameters and architecture identical to E1**, but replacing the loss with a Min-SNR-reweighted version, does FID@10k (NFE=50) and/or training efficiency improve?

---

- **Hypothesis / expectation:**  
  Based on prev results this baseline should be undertrained with a garbage fid again, so expecting such again but maybe timestep weighting brings FID down / at least improves model's later timesteps.

- **Primary comparison:**  
  E2 (Min-SNR linear, 50k) will be compared to E1 at matched steps (50k) and compute.

---

## Configuration

- **Config path:**  
  `configs/study/MS1_min_snr/e2/e2_minsnr_linear.yaml`

- **Params:**
  - Model: `unet_cifar32`
  - Dataset: CIFAR-10, 32×32, full train set
  - Beta schedule: `linear`
  - Diffusion steps: `num_timesteps: 1000` 
  - Loss: `minsnr_gamma: 5.0`
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
  - Final FID@10k either improves or is in the same ballpark as vanilla (190-ish i expect).


---

## DOD

Commands to run:
   ```bash
   python -m ablation_harness.cli run \
     --config configs/study/MS1_min_snr/configs/study/MS1_min_snr/e2/e2_minsnr_linear.yaml \
     --out_dir /content/drive/MyDrive/min-snr-e2/runs