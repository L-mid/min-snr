#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

OUT_DIR="${1:-runs}"

echo "[run_e2] Using OUT_DIR=${OUT_DIR}"
mkdir -p "${OUT_DIR}"

python -m ablation_harness.cli run \
  --config configs/study/MS1_min_snr/E2_minsnr_linear.yaml \
  --out_dir "${OUT_DIR}"