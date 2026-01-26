#!/usr/bin/env python3
"""
Radxa Cubie A7A System Optimizer
Tuning CPU, Memory, and Threads for 6GB Cortex-A76/A55 SoC
"""

import os
import sys
import subprocess
import json

LOG_FILE = "optimization_log.txt"

def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def run_cmd(cmd, shell=False):
    try:
        if shell:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        log(f"⚠️  Command failed: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
        log(f"   Error: {e.stderr.strip()}")
        return None

def is_root():
    return os.geteuid() == 0

def optimize_cpu():
    log("\n🏎️  Optimizing CPU Performance...")
    
    # Identify cores
    # Cubie A7A: 4x A76 + 4x A55 usually, or similar Cortex-A76/A55 mix
    log("ℹ️  CPU Info: Cores 0-1 (A76), Cores 2-7 (A55)")
    
    for i in range(8):
        gov_path = f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor"
        if os.path.exists(gov_path):
            try:
                with open(gov_path, "r") as f:
                    old_gov = f.read().strip()
                
                # Set to performance
                run_cmd(f"echo performance | sudo tee {gov_path}", shell=True)
                
                with open(gov_path, "r") as f:
                    new_gov = f.read().strip()
                
                log(f"   ✓ CPU {i}: {old_gov} → {new_gov}")
            except Exception as e:
                log(f"   ❌ Failed to set CPU {i} governor: {e}")

def optimize_memory():
    log("\n💾 Optimizing Memory Management (6GB RAM)...")
    
    configs = [
        ("vm.swappiness", "5"),
        ("vm.vfs_cache_pressure", "50"),
        ("vm.dirty_ratio", "10"),
        ("vm.dirty_background_ratio", "5")
    ]
    
    for param, value in configs:
        old_val = run_cmd(["sysctl", "-n", param])
        run_cmd(["sudo", "sysctl", "-w", f"{param}={value}"])
        new_val = run_cmd(["sysctl", "-n", param])
        log(f"   ✓ {param}: {old_val} → {new_val}")

def set_thread_envs():
    log("\n🧵 Setting Thread Optimization Envs...")
    
    envs = {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "VECLIB_MAXIMUM_THREADS": "2",
        "NUMEXPR_NUM_THREADS": "2"
    }
    
    log("ℹ️  Add these to your .bashrc or venv/bin/activate for persistence:")
    for k, v in envs.items():
        log(f"   export {k}={v}")
        os.environ[k] = v

def main():
    log(f"=== Radxa Cubie A7A Optimization Log ({subprocess.check_output(['date']).decode().strip()}) ===")
    
    if not is_root():
        log("⚠️  WARNING: Not running as root. CPU/Sysctl changes may fail.")
        log("   Please run: sudo python3 optimize_system.py")
        print("\n")

    optimize_cpu()
    optimize_memory()
    set_thread_envs()
    
    log("\n✅ Optimization Complete!")
    log("   Check optimization_log.txt for details.")

if __name__ == "__main__":
    main()
