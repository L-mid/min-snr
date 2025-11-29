# Min-SNR DDPM Study

This repo is a thin study wrapper around [`ablation-harness`](external/ablation-harness) for:
- Comparing **Min-SNR loss reweighting** vs a **vanilla ε-prediction DDPM** baseline.
- Running tightly controlled experiments on CIFAR-10 (32×32) with shared infra.
- Tracking preregistrations, results, and plots in one place.

Core logic (models, training loop, samplers, etc.) lives in `external/ablation-harness`.  
This repo mainly provides **configs, docs, and plots** for the Min-SNR project.

---

## Quickstart

### Clone with submodules

```bash
git clone --recurse-submodules https://github.com/L-mid/min-snr.git
cd min-snr
```

### If you already cloned without submodules:

```bash
git submodule update --init --recursive
```