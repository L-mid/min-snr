# [ID] – E2 Min-SNR vs baseline, linear β, 50k steps

## Question

On the same small, underpowered CIFAR-10 DDPM used in E1/E2, what does Min-SNR (γ=5) look like in the first 10k training steps if we just truncate the run?

If I take the E2 config (Min-SNR, γ=5, linear β, same model, same batch size, same optimizer, etc.) and only change total_steps from 50k → 10k, what does early FID@10k (NFE=50) look like?

And does the early training trajectory (FID vs step, loss vs step) look like a scaled-down version of the full 50k run (E2), or is it qualitatively different?

---

- **Hypothesis / expectation:**  
    FID will still be bad in absolute terms (undertrained baseline, tiny batch), but:

    Min-SNR (γ=5) might already show some improvement in FID / behaviour by 10k steps compared to the vanilla baseline, and

    the shape/order of the early FID curve in E3 should roughly match what I see in the full 50k E2 run (i.e. “good early” ≈ “good later” for this config).

    (also: this is an oppertunity to test my new stats wiring).


- **Primary comparison:**  
    Within Min-SNR: E3 (10k steps) vs E2 (50k steps) to see how early behaviour compares to the full run.


## Configuration

- **Config path:**  
  `configs/study/MS1_min_snr/e3/e3_minsnr_gamma5_short.yaml`

- **Params:**
  - Model: `unet_cifar32`
  - Dataset: CIFAR-10, 32×32, full train set
  - Beta schedule: `linear`
  - Diffusion steps: `num_timesteps: 1000` 
  - Loss: `minsnr_gamma: 5.0`
  - Sampler: `ddpm` at eval, `nfe: 50`
  - Total steps: `10_000`
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
  - Early FID checkpoints at steps {2k, 4k, 6k, 8k, 10k} to see the curve.

    - Training loss curve for sanity (check for divergence / weird spikes).

- **Success criteria:**
  - The run completes 10k steps without obvious divergence (loss NaNs, FID exploding, etc.).


---

## DOD

Commands to run:
   ```bash
   python -m ablation_harness.cli run \
     --config configs/study/MS1_min_snr/configs/study/MS1_min_snr/e3/e3_minsnr_gamma5_short.yaml \
     --out_dir /content/drive/MyDrive/min-snr-e3/runs