from pathlib import Path
import subprocess, sys

def test_data_prep_cli(tmp_path: Path):
    out = tmp_path / "data.csv"
    cmd = [sys.executable, "src/data_prep.py", "--rows", "100", "--seed", "42", "--out", str(out)]
    subprocess.check_call(cmd)
    assert out.exists() and out.stat().st_size > 0
