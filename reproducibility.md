## REM from C:\Users\grayd\min-snr

REM 1) Create venv (once)
python -m venv .venv

REM 2) Activate venv (each new terminal)
.\.venv\Scripts\activate

REM 3) Upgrade pip (optional)
python -m pip install --upgrade pip

REM 4) Install ablation-harness in editable mode
python -m pip install -e external\ablation-harness



