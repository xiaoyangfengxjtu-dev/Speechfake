#!/usr/bin/env python3
"""
Check server configuration and dataset paths
"""

import json
import os
from pathlib import Path

def check_dataset_paths(config_dir: str):
    """
    Check if dataset paths in config files exist
    """
    config_path = Path(config_dir)
    conf_files = list(config_path.rglob("*.conf"))
    
    print("Dataset Path Configuration Check")
    print("=" * 80)
    
    issues = []
    valid_paths = []
    
    for conf_file in conf_files:
        try:
            with open(conf_file, 'r') as f:
                config = json.load(f)
            
            if 'database_path' in config:
                db_path = config['database_path']
                path_exists = os.path.exists(db_path)
                
                status = "EXISTS" if path_exists else "MISSING"
                print(f"{status:8} {conf_file}")
                print(f"         Path: {db_path}")
                
                if path_exists:
                    valid_paths.append(db_path)
                    
                    # Check subdirectories for datasets
                    subdirs = ['ASVspoof2019_LA', 'SPEECHFAKE', 'ITW', 'FOR']
                    for subdir in subdirs:
                        subdir_path = os.path.join(db_path, subdir)
                        subdir_exists = os.path.exists(subdir_path)
                        subdir_status = "EXISTS" if subdir_exists else "MISSING"
                        print(f"         {subdir}: {subdir_status}")
                else:
                    issues.append(f"Path not found: {db_path} in {conf_file}")
                
                print()
                
        except Exception as e:
            issues.append(f"Error reading {conf_file}: {e}")
    
    # Summary
    print("=" * 80)
    print("Summary:")
    print(f"  Configuration files checked: {len(conf_files)}")
    print(f"  Unique dataset paths: {len(set(valid_paths))}")
    print(f"  Issues found: {len(issues)}")
    
    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"  - {issue}")
    
    return len(issues) == 0

def check_required_datasets(base_path: str):
    """
    Check if required datasets are present
    """
    print("\nRequired Dataset Check")
    print("=" * 80)
    
    required_datasets = {
        'ASVspoof2019_LA': [
            'ASVspoof2019_LA_train',
            'ASVspoof2019_LA_dev', 
            'ASVspoof2019_LA_eval',
            'ASVspoof2019_LA_cm_protocols'
        ],
        'SPEECHFAKE': [
            'train',
            'dev',
            'test'
        ],
        'ITW': [],
        'FOR': []
    }
    
    all_good = True
    
    for dataset, subdirs in required_datasets.items():
        dataset_path = os.path.join(base_path, dataset)
        
        if os.path.exists(dataset_path):
            print(f"EXISTS  {dataset}")
            
            for subdir in subdirs:
                subdir_path = os.path.join(dataset_path, subdir)
                if os.path.exists(subdir_path):
                    print(f"  EXISTS  {subdir}")
                else:
                    print(f"  MISSING {subdir}")
                    all_good = False
        else:
            print(f"MISSING {dataset}")
            all_good = False
    
    return all_good

def main():
    # Check configuration files
    config_ok = check_dataset_paths(".")
    
    # Check specific dataset path (you can modify this)
    dataset_base = "/path/to/your/datasets"  # Modify this path
    
    if os.path.exists(dataset_base):
        datasets_ok = check_required_datasets(dataset_base)
    else:
        print(f"\nDataset base path not found: {dataset_base}")
        print("Please update the dataset_base variable in this script")
        datasets_ok = False
    
    print("\n" + "=" * 80)
    print("Overall Status:")
    print(f"  Configuration: {'OK' if config_ok else 'ISSUES FOUND'}")
    print(f"  Datasets: {'OK' if datasets_ok else 'ISSUES FOUND'}")
    
    if config_ok and datasets_ok:
        print("\nReady to run experiments!")
    else:
        print("\nPlease fix the issues above before running experiments.")

if __name__ == "__main__":
    main()
