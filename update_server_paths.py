#!/usr/bin/env python3
"""
Update database paths for server deployment
Usage: python update_server_paths.py --old_path "./datasets/" --new_path "/path/to/server/datasets/"
"""

import json
import os
import argparse
from pathlib import Path

def update_config_paths(config_dir: str, old_path: str, new_path: str, dry_run: bool = False):
    """
    Update database_path in all .conf files
    """
    config_path = Path(config_dir)
    conf_files = list(config_path.rglob("*.conf"))
    
    updated_files = []
    
    print(f"Searching for configuration files in: {config_path}")
    print(f"Old path: {old_path}")
    print(f"New path: {new_path}")
    print(f"Dry run: {dry_run}")
    print("-" * 80)
    
    for conf_file in conf_files:
        try:
            with open(conf_file, 'r') as f:
                config = json.load(f)
            
            # Check if database_path needs updating
            if 'database_path' in config:
                current_path = config['database_path']
                
                # Update if it matches old_path or is a relative path
                should_update = (
                    current_path == old_path or 
                    current_path.startswith('./') or
                    current_path == "/path/to/datasets/"
                )
                
                if should_update:
                    if not dry_run:
                        config['database_path'] = new_path
                        
                        with open(conf_file, 'w') as f:
                            json.dump(config, f, indent=4)
                    
                    updated_files.append({
                        'file': str(conf_file),
                        'old_path': current_path,
                        'new_path': new_path
                    })
                    
                    status = "[DRY RUN]" if dry_run else "[UPDATED]"
                    print(f"{status} {conf_file}")
                    print(f"    {current_path} -> {new_path}")
                
        except Exception as e:
            print(f"Error processing {conf_file}: {e}")
    
    return updated_files

def update_specific_configs(old_path: str, new_path: str, dry_run: bool = False):
    """
    Update specific configuration files used in Table 3 experiments
    """
    config_files = [
        # Table 3 experiment configs
        "experiments/table3_aasist_asv19.conf",
        "experiments/table3_aasist_bd.conf", 
        "experiments/table3_aasist_bd_en.conf",
        "experiments/table3_aasist_bd_cn.conf",
        "experiments/table3_w2v_asv19.conf",
        "experiments/table3_w2v_bd.conf",
        "experiments/table3_w2v_bd_en.conf",
        "experiments/table3_w2v_bd_cn.conf",
        
        # Other important configs
        "experiments/paper_exact_reproduction.conf",
        "experiments/exp_bd_w2v_aasist.conf",
        "config/AASIST.conf",
        "config/SpeechFake_W2V_AASIST.conf"
    ]
    
    updated_files = []
    
    print("Updating specific configuration files:")
    print("-" * 80)
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                if 'database_path' in config:
                    current_path = config['database_path']
                    
                    # Always update these specific files
                    if not dry_run:
                        config['database_path'] = new_path
                        
                        with open(config_file, 'w') as f:
                            json.dump(config, f, indent=4)
                    
                    updated_files.append({
                        'file': config_file,
                        'old_path': current_path,
                        'new_path': new_path
                    })
                    
                    status = "[DRY RUN]" if dry_run else "[UPDATED]"
                    print(f"{status} {config_file}")
                    print(f"    {current_path} -> {new_path}")
                    
            except Exception as e:
                print(f"Error processing {config_file}: {e}")
        else:
            print(f"File not found: {config_file}")
    
    return updated_files

def main():
    parser = argparse.ArgumentParser(description="Update database paths for server deployment")
    parser.add_argument("--old_path", type=str, default="./datasets/",
                       help="Old database path to replace")
    parser.add_argument("--new_path", type=str, required=True,
                       help="New database path for server")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be changed without making changes")
    parser.add_argument("--specific_only", action="store_true",
                       help="Only update specific Table 3 experiment configs")
    
    args = parser.parse_args()
    
    print("Server Path Update Script")
    print("=" * 80)
    
    if args.specific_only:
        updated_files = update_specific_configs(args.old_path, args.new_path, args.dry_run)
    else:
        updated_files = update_config_paths(".", args.old_path, args.new_path, args.dry_run)
    
    print("\n" + "=" * 80)
    print(f"Summary:")
    print(f"  Files processed: {len(updated_files)}")
    
    if args.dry_run:
        print("  Mode: DRY RUN (no changes made)")
        print("  To apply changes, run without --dry_run flag")
    else:
        print("  Mode: UPDATES APPLIED")
    
    print("\nNext steps:")
    print("1. Verify the new paths are correct")
    print("2. Ensure datasets are available at the new location")
    print("3. Test with a small experiment to verify configuration")

if __name__ == "__main__":
    main()
