import os
from pathlib import Path
import subprocess
import sys


TESTS = (
    "vbar-npu.py",
    "vbar-control-npu.py",
    "alloc-npu.py",
    "va-release-npu.py",
    "malloc-graph-small-npu.py",
    "malloc-graph-control-npu.py",
    "host-buffer-npu.py",
    "hostbuf-slice-npu.py",
    "multi-device-npu.py",
)

directory = Path(__file__).parent
env = os.environ | {"PYTHONPATH": str(directory.parent)}
for test in TESTS:
    subprocess.run([sys.executable, directory / test], env=env, check=True)

print(f"All {len(TESTS)} NPU tests passed")
