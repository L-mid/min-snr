# E24 — Tiny robustness / failure-mode: does Min-SNR break under early-t training?

## Question
Does Min-SNR implicitly rely on uniform timestep sampling, and does it collapse (behaviorally or numerically) when training mass is concentrated at very small t (high SNR)?

## Motivation
Min-SNR downweights high-SNR steps. If we train only on early timesteps (t≈0), weights may become tiny and:
- (A) learning signal collapses (intended behavior: “don’t learn from high SNR”), or
- (B) weights underflow to zero under AMP (numeric failure), causing true training freeze.

This is a direct stress test for an implicit assumption: training over the full t range.

## Setup (shared)
- Dataset: CIFAR-10
- Model: unet_cifar32, base_channels=64
- Beta schedule: cosine
- Optim: Adam lr=1e-4
- EMA: 0.999
- Steps: 3000
- Seeds: 2222, 3333, 4444
- Timestep sampling: **uniform over t ∈ [0, 50]** (early-t only)
- Eval: recon + small grids (KID/FID milestone off; this is a failure-mode probe)

## Variants
A. Constant loss weighting (early-t only)
B. Min-SNR weighting, γ=5 (early-t only)
C. (Optional) Same as B but AMP off (numeric underflow disambiguation)

## Hypotheses
H1 (behavioral): B learns significantly slower or appears stalled vs A because Min-SNR heavily downweights high SNR steps.
H2 (numeric): Under AMP, B shows weight underflow (weights become exact 0), causing near-zero gradients; turning AMP off (C) restores training.

## Metrics to inspect (must-have)
Training-side:
- loss curve (train/*)
- grad_norm (if logged)
- (if available) Min-SNR weight stats over batch: mean / min / p01 / p50 / p99 / zero-fraction
Evaluation-side:
- recon mse/psnr at fixed t in-range: t=[10,25,50]
- sample grids at NFE=20 (DDIM) at steps 1k/2k/3k

## Analysis plan
Primary comparison: A vs B under the same timestep distribution.
- If B’s loss/recon improves far less than A and/or gradients collapse early, mark: Min-SNR breaks under early-t curriculum.
- If AMP-off (C) fixes collapse (vs B), classify as numeric failure (fp16 underflow).
- If AMP-off does not fix collapse, classify as behavioral failure (loss reweighting removes learning signal in this regime).

## Decision / write-up note
This is expected to be small and fast. Outcome feeds the final report as:
- Min-SNR assumes broad timestep coverage; truncated t distributions can stall training.
- If numeric: AMP exacerbates Min-SNR underflow at high SNR; mitigate via fp32 weights/clamping/log-weighting.
