import json
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(SCRIPT_DIR, ".venv", "Scripts", "python.exe")
BENCHMARK_SCRIPT = os.path.join(SCRIPT_DIR, "scripts", "tiny_recall_benchmark.py")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "benchmark_results.json")

# Use current python (will be conda env if activated), fall back to .venv
PYTHON_EXE = sys.executable
if not os.path.exists(VENV_PYTHON):
    pass  # stick with sys.executable
else:
    PYTHON_EXE = VENV_PYTHON

cmd = [
    PYTHON_EXE,
    BENCHMARK_SCRIPT,
    "--device", "cpu",
    "--steps", "100",
    "--batch-size", "8",
    "--eval-episodes", "64",
    "--seed", "42",
]

print(f"Running: {' '.join(cmd)}")

result = subprocess.run(
    cmd,
    cwd=SCRIPT_DIR,
    capture_output=True,
    text=True,
    timeout=600,
)

stdout = result.stdout.strip()
stderr = result.stderr.strip()

print("=== STDOUT ===")
print(stdout)
if stderr:
    print("=== STDERR ===")
    print(stderr)

succeeded = (result.returncode == 0)

try:
    benchmark_data = json.loads(stdout)
except json.JSONDecodeError:
    last_lines = stdout.splitlines()
    for line in reversed(last_lines):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                benchmark_data = json.loads(line)
                break
            except json.JSONDecodeError:
                pass
    else:
        benchmark_data = None

output = {
    "succeeded": succeeded,
    "gpt_baseline": benchmark_data.get("gpt_baseline") if benchmark_data else None,
    "then_reset": benchmark_data.get("then_reset") if benchmark_data else None,
    "then_persistent": benchmark_data.get("then_persistent") if benchmark_data else None,
    "then_shuffled_state": benchmark_data.get("then_shuffled_state") if benchmark_data else None,
    "stdout": stdout,
    "stderr": stderr,
    "returncode": result.returncode,
}

with open(RESULTS_FILE, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {RESULTS_FILE}")
print(json.dumps({k: v for k, v in output.items() if k in ("succeeded", "gpt_baseline", "then_reset", "then_persistent", "then_shuffled_state", "returncode")}, indent=2))

if not succeeded:
    sys.exit(result.returncode)