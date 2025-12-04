# Environment setup for Min-SNR study

This repo is a thin wrapper around `external/ablation-harness`.  
All core training logic and dependencies live in that submodule.

The goal of this document is to say **exactly what environment** was used when running the experiments.

---

## 1. Python & OS

- Python: 3.11.x (CPython)
- OS: (e.g.) Windows 10 with WSL2 / Ubuntu 22.04, or native Linux
- CUDA: (if applicable) 11.x with a compatible Google Colab GPU

---

## 2. Local setup

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows powershell: .venv\Scripts\activate
pip install --upgrade pip
pip install -e external/ablation-harness

# for tests + plots or if you want to edit stuff:
pip install -e ".[dev]"
```

## TORCH PINNED:
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121
```


## 3. Colab setup 

Create a new Colab notebook.
Cells (together or disjointed):

```bash

# Mount Google Drive (optional, if you want to save runs):
from google.colab import drive
drive.mount("/content/drive")


# Clone this repo with submodules:
%cd /content
!git clone --recurse-submodules https://github.com/L-mid/min-snr.git
%cd min-snr


# install dependencies:
!python -m pip install --upgrade pip
!python -m pip install -e external/ablation-harness

# ensure to reinstall pinned torch for intended behaviour:
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121


# run a baseline!:
!python -m ablation_harness.cli run \
    --config configs/study/MS1_min_snr/e1/e1_baseline_linear.yaml \
    --out_dir runs


```


### Note:

GPU vs CPU:

**GPU is strongly recommended for full experiments.**

For smoke tests (short runs, CI, etc.), CPU is sufficient:

- Use E1_smoke_linear.yaml with total_steps=5 for simple smoke test.
- ema.enabled = false and amp = false to avoid GPU-specific paths on cpu in yaml.
