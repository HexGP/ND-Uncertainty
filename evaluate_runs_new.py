#!/usr/bin/env python3
"""
Simple evaluation script for runs_new using evaluate_single_scene.py
"""

import os
import subprocess
import argparse
from pathlib import Path

scans = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]

def find_latest_experiment(runs_dir, exp_name, scan_idx):
    """Find the most recent experiment directory for a scan."""
    exp_pattern = f"{exp_name}_{scan_idx}"
    exp_dir = os.path.join(runs_dir, exp_pattern)
    
    if not os.path.exists(exp_dir):
        return None, None
    
    # Look for timestamp directories
    dirs = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
    if not dirs:
        return None, None
    
    # Sort by timestamp (format: YYYY-MM-DD_HH-MM-SS)
    from datetime import datetime
    import re
    timestamp_dirs = [d for d in dirs if re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', d)]
    if timestamp_dirs:
        timestamp_dirs.sort(key=lambda x: datetime.strptime(x, '%Y-%m-%d_%H-%M-%S'))
        latest_timestamp = timestamp_dirs[-1]
        exp_path = os.path.join(exp_dir, latest_timestamp)
    else:
        # No timestamp, use the directory itself
        exp_path = exp_dir
    
    # Find latest mesh
    plots_dir = os.path.join(exp_path, 'plots')
    if not os.path.exists(plots_dir):
        return None, None
    
    import glob
    mesh_files = glob.glob(os.path.join(plots_dir, "mesh_*.ply"))
    if not mesh_files:
        return None, None
    
    # Sort by epoch number
    mesh_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]) if os.path.basename(x).split('_')[1].split('.')[0].isdigit() else 0)
    latest_mesh = mesh_files[-1]
    
    return exp_path, latest_mesh

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate all runs_new using evaluate_single_scene.py')
    parser.add_argument('--runs_dir', type=str, default='runs_new', help='Root directory containing runs')
    parser.add_argument('--data_dir', type=str, default='./data/Replica', help='Path to Replica dataset')
    parser.add_argument('--exp_name', type=str, default='replica', help='Experiment name pattern')
    parser.add_argument('--output_dir', type=str, default='evaluation_results_new', help='Output directory')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    eval_script = os.path.join(script_dir, 'evals', 'replica_eval', 'evaluate_single_scene.py')
    
    if not os.path.exists(eval_script):
        print(f"Error: {eval_script} not found")
        exit(1)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Evaluating all scans in runs_new...")
    print("="*80)
    
    for idx, scan in enumerate(scans):
        scan_idx = idx + 1
        print(f"\nProcessing {scan} (scan{scan_idx})...")
        
        exp_path, mesh_file = find_latest_experiment(args.runs_dir, args.exp_name, scan_idx)
        
        if mesh_file is None:
            print(f"  No mesh found for {scan}")
            continue
        
        print(f"  Found mesh: {mesh_file}")
        
        # Convert to absolute paths for Windows compatibility
        mesh_file_abs = os.path.abspath(mesh_file)
        data_dir_abs = os.path.abspath(args.data_dir)
        output_dir_abs = os.path.abspath(args.output_dir)
        
        # Run evaluate_single_scene.py
        cmd = f"python {eval_script} --input_mesh {mesh_file_abs} --scan_id {scan_idx} --data_dir {data_dir_abs} --output_dir {output_dir_abs}"
        print(f"  Running: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"  [OK] Successfully evaluated {scan}")
            if result.stdout:
                # Print last few lines of output
                lines = result.stdout.strip().split('\n')
                for line in lines[-3:]:
                    if line.strip():
                        print(f"    {line}")
        else:
            print(f"  [FAILED] Failed to evaluate {scan}")
            if result.stderr:
                error_msg = result.stderr
                # Show more of the error
                if len(error_msg) > 500:
                    print(f"    Error (first 500 chars): {error_msg[:500]}")
                    print(f"    Error (last 200 chars): {error_msg[-200:]}")
                else:
                    print(f"    Error: {error_msg}")
            if result.stdout:
                # Also check stdout for errors
                stdout_lines = result.stdout.strip().split('\n')
                error_lines = [l for l in stdout_lines if 'Error' in l or 'Traceback' in l or 'Exception' in l]
                if error_lines:
                    print(f"    Output errors: {error_lines[-3:]}")
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print(f"Results saved to: {os.path.join(args.output_dir, 'evaluation_results.txt')}")
    print("="*80)
