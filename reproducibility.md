# Reproducibility checklist

This document records the minimal information needed to reproduce the Min-SNR experiments.

---

## 1. Seeds and determinism

- Global training seed: `1077` for full E1 / E2 runs (baseline vs Min-SNR).
- Smoke test seed: `1234` in `E1_smoke_linear.yaml`.
- `deterministic: true` is enabled in configs to make runs as stable as practical.

Note: Perfect determinism across different hardware / CUDA / driver versions
is not guaranteed, but these settings are sufficient for stable comparisons on
the same setup.

---

## 2. Submodule commit (ablation-harness)

The `external/ablation-harness` submodule is pinned to a specific commit.

To see which one:

```bash
cd external/ablation-harness
git rev-parse HEAD
```