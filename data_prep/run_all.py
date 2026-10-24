"""Run the whole data-prep pipeline: python run_all.py"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["01_clean_base.py", "02_build_labels.py", "03_inject.py", "04_build_accounts.py", "05_validate.py"]

for step in STEPS:
    print(f"\n{'=' * 8} {step} {'=' * 8}", flush=True)
    t = time.time()
    if subprocess.run([sys.executable, str(HERE / step)], cwd=HERE).returncode != 0:
        sys.exit(f"{step} failed - fix it and re-run (earlier steps do not need repeating).")
    print(f"-- {step} done in {time.time() - t:.0f}s", flush=True)
print("\nPipeline complete. Final files are in data/final/")
