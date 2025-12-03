import subprocess
import sys
from pathlib import Path


def test_cli_smoke_e1_baseline(tmp_path):
    """
    Smoke test: run a tiny Min-SNR baseline config for a few steps
    and verify that some logs are written under runs_test/.
    """
    project_root = Path(__file__).resolve().parents[1]
    runs_root = project_root / "runs_test" / "cli_smoke"

    # Clean up old runs if they exist
    if runs_root.exists():
        # best-effort cleanup
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
        "configs/study/MS1_min_snr/e1/e1_smoke_linear.yaml",
        "--out_dir",
        str(tmp_path),
    ]

    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)

    assert result.returncode == 0, (
        f"CLI run failed with code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
    )

    assert tmp_path.exists(), "Expected runs_root directory to exist after run"

    jsonl_files = list(tmp_path.rglob("*.jsonl"))
    assert jsonl_files, (
        "Expected at least one .jsonl log file under runs_test/cli_smoke, "
        "but found none"
    )