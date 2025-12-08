# W1-E5 – Hutchinson trace probe (linear baseline)

## Summary

- **Question:**  
For this small CIFAR-10 DDPM baseline, how does a simple curvature proxy (Hutchinson trace estimate of the loss Hessian) evolve over training? And how does it relate to training loss and FID?

- **Hypothesis / expectation:**  
    Early in training, the Hutchinson trace should be relatively large (high curvature), then gradually decrease as the model moves into flatter regions.

    Spikes in the trace should roughly co-occur with loss spikes / instabilities (if any).

    Even if FID is bad in absolute terms, we expect qualitative structure: e.g. curvature decays fastest in the first few thousand steps and then plateaus.

    This run is mainly about getting usable, stable curvature traces without exploding compute; we don’t necessarily expect big FID differences vs E1.

- **Primary comparison:**  
  Loss / FID behaviour vs E1 baseline (same architecture/hparams, just with extra logging and less steps).

---

## Configuration

- **Config path:**  
  `configs/study/MS1_min_snr/e5/e5_baseline_hutch_trace.yaml`

- **Key knobs:**
  - Model: `CIFAR-10, 32×32`
  - Beta schedule: `linear`
  - Loss / weighting: `constant`    (none)
  - Sampler + NFE: `DDPM, 50`
  - Total steps: `10000`
  - Batch size: `4`
  - EMA: `0.999`

**Hutchinson trace settings (new for E5):**
    - Probes: 8–16 random Rademacher vectors per measurement.

    - Measurement frequency: every 100 global steps.

    - Logged keys: curvature/hutch_trace_mean, curvature/hutch_trace_std, maybe raw per-probe if feasable. 

- **Seed(s):**
  - Train seed: `1077`

---

## Metrics & success criteria

- **Primary metric(s):**
  - Training loss vs step (same as E1).
  - FID@{2k, 4k, 6k, 8k, 10k} (NFE=50)
  - Hutchinson trace estimate over training

- **Success criteria:**
  - This is mostly an instrumentation / probe experiment; success is about getting a clean signal, not beating anything yet
  - Will be used for comparasions later.

---

## Command(s) to run:
   ```bash
    python -m ablation_harness.cli run \
    --config configs/study/MS1_min_snr/e5/e5_baseline_hutch_trace.yaml \
    --out_dir runs/MS1_min_snr/       e5_baseline_hutch_trace   # example out_dir