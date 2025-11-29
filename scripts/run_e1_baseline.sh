
# Cool script
# Now you can:

# bash scripts/run_e1_baseline.sh runs/E1_baseline
# bash scripts/run_e2_minsnr.sh runs/E2_minsnr

# for example


#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (directory containing this script's parent)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Optional first argument: custom out_dir (default: runs)
OUT_DIR="${1:-runs}"

echo "[run_e1] Using OUT_DIR=${OUT_DIR}"
mkdir -p "${OUT_DIR}"

python -m ablation_harness.cli run \
  --config configs/study/MS1_min_snr/E1_baseline_linear.yaml \
  --out_dir "${OUT_DIR}"