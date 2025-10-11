#!/usr/bin/env python3
"""
Batch script to run all Table 3 experiments
"""

import subprocess
import sys
from pathlib import Path
import time

def run_experiment(config_file: str, seed: int = 1234, comment: str = ""):
    """
    Run a single experiment
    """
    cmd = [
        sys.executable, "main.py",
        "--config", config_file,
        "--seed", str(seed),
        "--output_dir", "./exp_result"
    ]
    
    if comment:
        cmd.extend(["--comment", comment])
    
    print(f"Running: {' '.join(cmd)}")
    print("-" * 80)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f" Completed: {config_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f" Failed: {config_file} (exit code: {e.returncode})")
        return False

def main():
    """
    Run all Table 3 experiments
    """
    experiments = [
        # AASIST experiments
        ("experiments/table3_aasist_asv19.conf", "AASIST_ASV19"),
        ("experiments/table3_aasist_bd.conf", "AASIST_BD"),
        ("experiments/table3_aasist_bd_en.conf", "AASIST_BD_EN"),
        ("experiments/table3_aasist_bd_cn.conf", "AASIST_BD_CN"),
        
        # W2V+AASIST experiments
        ("experiments/table3_w2v_asv19.conf", "W2V_ASV19"),
        ("experiments/table3_w2v_bd.conf", "W2V_BD"),
        ("experiments/table3_w2v_bd_en.conf", "W2V_BD_EN"),
        ("experiments/table3_w2v_bd_cn.conf", "W2V_BD_CN"),
    ]
    
    print("Starting Table 3 Overall Performance Experiments")
    print("=" * 80)
    print(f"Total experiments: {len(experiments)}")
    print("=" * 80)
    
    successful = 0
    failed = 0
    
    for i, (config_file, comment) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Starting: {comment}")
        
        if run_experiment(config_file, seed=1234, comment=comment):
            successful += 1
        else:
            failed += 1
        
        print(f"Progress: {successful + failed}/{len(experiments)} completed")
        print("=" * 80)
    
    print(f"\nFinal Results:")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {successful/len(experiments)*100:.1f}%")
    
    if successful == len(experiments):
        print("\nAll experiments completed successfully!")
        print("You can now run: python table3_evaluation.py")
    else:
        print(f"\n{failed} experiments failed. Check logs for details.")

if __name__ == "__main__":
    main()
