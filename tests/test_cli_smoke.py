import subprocess
import sys
from pathlib import Path


import subprocess
import sys
from pathlib import Path


def _run_cli_and_check(config_path: str, out_dir: str):
    project_root = Path(__file__).resolve().parents[1]
    runs_root = project_root / out_dir

    # Clean up old runs if they exist
    if runs_root.exists():
        for p in sorted(runs_root.rglob("*"), reverse=True):
            try:
                p.unlink()
            except IsADirectoryError:
                p.rmdir()
        runs_root.rmdir()

    cmd = [
        sys.executable,
        "-m",
        "ablation_harness.cli",
        "run",
        "--config",
        config_path,
        "--out_dir",
        str(runs_root),
    ]

    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

    assert result.returncode == 0, (
        f"CLI run failed with code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    assert runs_root.exists(), "Expected runs_root directory to exist after run"

    jsonl_files = list(runs_root.rglob("*.jsonl"))
    assert jsonl_files, (
        f"Expected at least one .jsonl log file under {runs_root}, "
        "but found none"
    )


def test_cli_smoke_e1_baseline(tmp_path):
    _run_cli_and_check(
        "configs/study/MS1_min_snr/e1/e1_smoke_linear.yaml",
        tmp_path,   # switch outdir to tangible for debugging
    )


def test_cli_smoke_e2_minsnr(tmp_path):
    _run_cli_and_check(
        "configs/study/MS1_min_snr/e2/e2_smoke_linear.yaml",
        tmp_path,
    )