"""
LiveMem THEN — Full Validation Runner (GPU Auto-Detect)
=========================================================
Environment check → Tests → Benchmark → Results

Usage:
    conda activate companion    # or your GPU-ready env
    cd F:\projects\LiveMem\nanochat-then
    python run_all.py

All results saved to ./_results/ directory.
"""
import json
import subprocess
import sys
import os
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "_results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "results.json"

results = {
    "timestamp": datetime.now().isoformat(),
    "python": sys.executable,
    "version": sys.version,
    "hardware": {},
    "environment": {},
    "tests": {},
    "benchmarks": {},
}


def run_step(name, cmd_args, timeout=600):
    """Run a command, capture output, return success + parsed data."""
    print(f"\n{'='*60}")
    print(f"  [{name}]")
    print(f"  {' '.join(cmd_args)}")
    print(f"{'='*60}\n")
    
    try:
        proc = subprocess.run(
            cmd_args,
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return {"exit_code": -1, "stdout": "", "stderr": f"Timeout after {timeout}s"}
    except FileNotFoundError:
        print(f"  COMMAND NOT FOUND: {cmd_args[0]}")
        return {"exit_code": -2, "stdout": "", "stderr": f"Command not found: {cmd_args[0]}"}
    
    stdout = proc.stdout
    stderr = proc.stderr
    print(stdout)
    if stderr:
        print(f"  [STDERR]\n{stderr}")
    
    data = {"exit_code": proc.returncode, "stdout": stdout, "stderr": stderr}
    
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data["parsed"] = json.loads(line)
                break
            except json.JSONDecodeError:
                pass
    
    return data


# ============================================================
# STEP 0: Hardware & Environment Check
# ============================================================
print("=" * 60)
print("  LiveMem THEN — Full Validation Runner")
print(f"  Python:  {sys.executable}")
print(f"  CWD:     {os.getcwd()}")
print("=" * 60)

print("\n[STEP 0] Probing hardware...")
hw_check = run_step("hardware_check", [
    sys.executable, "-c", """
import json, torch, sys

cuda_ok = torch.cuda.is_available()
info = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "cuda_available": cuda_ok,
    "cuda_version": torch.version.cuda if cuda_ok else "N/A",
    "gpu_count": torch.cuda.device_count() if cuda_ok else 0,
    "cpu_count": torch.get_num_threads(),
}

if cuda_ok:
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info[f"gpu_{i}"] = {
            "name": props.name,
            "vram_gb": round(props.total_mem / 1024**3, 1),
            "compute_capability": f"{props.major}.{props.minor}",
        }

print(json.dumps(info, indent=2))
"""
])
results["hardware"] = hw_check

hw_parsed = hw_check.get("parsed", {})
has_cuda = hw_parsed.get("cuda_available", False)
DEVICE = "cuda" if has_cuda else "cpu"

if has_cuda:
    gpu_name = hw_parsed.get("gpu_0", {}).get("name", "Unknown GPU")
    gpu_vram = hw_parsed.get("gpu_0", {}).get("vram_gb", "?")
    print(f"\n  >>> GPU DETECTED: {gpu_name} ({gpu_vram} GB VRAM)")
    print(f"  >>> Using --device cuda for benchmarks\n")
else:
    print("\n  !!! NO CUDA DETECTED — check you activated the right conda env")
    print("  !!! Try: conda activate companion (or your GPU-ready env)\n")

# nanochat import check
import_check = run_step("nanochat_import", [
    sys.executable, "-c", """
import json
try:
    from nanochat.gpt import GPT, GPTConfig, THENGPT
    from nanochat.memory_manager import DiskTieredMemory
    from nanochat.common import COMPUTE_DTYPE
    print(json.dumps({
        "nanochat_import": "ok",
        "COMPUTE_DTYPE": str(COMPUTE_DTYPE),
    }))
except Exception as e:
    print(json.dumps({
        "nanochat_import": "FAILED",
        "error": str(e),
    }))
    raise
"""
])
results["nanochat_import"] = import_check

if import_check.get("exit_code", 1) != 0:
    print("\n  !!! nanochat import FAILED — stopping")
    print("  !!! Make sure you're in F:\\projects\\LiveMem\\nanochat-then")
    sys.exit(1)

# ============================================================
# STEP 1: Run Unit Tests
# ============================================================
print("\n\n[STEP 1] Running unit tests...")
tests = run_step("unit_tests", [
    sys.executable, "-m", "pytest", "tests/test_live_memory.py", "-v", "--tb=short",
    "-p", "no:warnings",
], timeout=120)
results["tests"] = tests

# ============================================================
# STEP 2: Run Benchmark — Quick Smoke Test
# ============================================================
print(f"\n\n[STEP 2] Quick benchmark ({DEVICE}, 200 steps)...")
bench_quick = run_step("benchmark_quick", [
    sys.executable, "scripts/tiny_recall_benchmark.py",
    "--device", DEVICE,
    "--steps", "200",
    "--batch-size", "16",
    "--eval-episodes", "128",
    "--seed", "42",
], timeout=300)
results["benchmarks"]["quick_200steps"] = bench_quick

# ============================================================
# STEP 3: Run Benchmark — Full Training
# ============================================================
print(f"\n\n[STEP 3] Full benchmark ({DEVICE}, 1000 steps)...")
bench_full = run_step("benchmark_full", [
    sys.executable, "scripts/tiny_recall_benchmark.py",
    "--device", DEVICE,
    "--steps", "1000",
    "--batch-size", "32",
    "--eval-episodes", "512",
    "--seed", "42",
], timeout=600)
results["benchmarks"]["full_1000steps"] = bench_full

# ============================================================
# SAVE RESULTS
# ============================================================
with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"  ALL DONE")
print(f"  Results: {RESULTS_FILE}")
print(f"{'='*60}")

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\n--- SUMMARY ---")
print(f"Device:       {DEVICE.upper()}")
if has_cuda:
    print(f"GPU:          {gpu_name} ({gpu_vram} GB)")
print(f"Python:       {hw_parsed.get('python', '?')}")
print(f"PyTorch:      {hw_parsed.get('torch', '?')}")
print(f"Unit tests:   {'PASS' if tests.get('exit_code') == 0 else 'FAIL'}")
print(f"Bench (200):  {'OK' if bench_quick.get('exit_code') == 0 else 'FAIL'}")
print(f"Bench (1000): {'OK' if bench_full.get('exit_code') == 0 else 'FAIL'}")

print("\n--- RECALL ACCURACY ---")
for label, bench in [("200 steps", bench_quick), ("1000 steps", bench_full)]:
    p = bench.get("parsed", {})
    if p:
        gpt    = p.get("gpt_baseline", "?")
        reset  = p.get("then_reset", "?")
        pers   = p.get("then_persistent", "?")
        shuff  = p.get("then_shuffled_state", "?")
        
        print(f"\n  {label}:")
        print(f"    gpt_baseline       = {gpt}")
        print(f"    then_reset         = {reset}")
        print(f"    then_persistent    = {pers}")
        print(f"    then_shuffled      = {shuff}")
        
        if isinstance(pers, (int, float)) and isinstance(reset, (int, float)):
            delta = pers - reset
            if delta > 0.01:
                print(f"    >>> PERSISTENT WINS: +{delta:.3f} over reset  ✅")
            elif delta > 0.001:
                print(f"    >>> MARGINAL: +{delta:.4f} — borderline")
            else:
                print(f"    >>> NO SIGNAL: persistent did not beat reset  ❌")
                print(f"    >>> Decision: PAUSE — strengthen task or debug traces before scaling")
    else:
        print(f"\n  {label}: [no results parsed — check output above]")