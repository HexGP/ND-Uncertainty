#!/usr/bin/env python3
"""
Script to evaluate scans 1, 2, and 3 from runs_unc_beta (first iteration, likely no max clamp).
Automatically detects latest mesh files and evaluates them.
"""
import os
import sys
import glob
import subprocess
from pathlib import Path

def find_latest_mesh(runs_dir, scan_id):
    """Find the latest mesh_*.ply file for a given scan."""
    scan_dir = os.path.join(runs_dir, f"replica_all8_{scan_id}")
    
    if not os.path.exists(scan_dir):
        print(f"[ERROR] Directory not found: {scan_dir}")
        return None
    
    # Find all experiment directories
    exp_dirs = [d for d in os.listdir(scan_dir) if os.path.isdir(os.path.join(scan_dir, d))]
    if not exp_dirs:
        print(f"[ERROR] No experiment directories found in {scan_dir}")
        return None
    
    # Sort by modification time (newest first)
    exp_dirs_with_time = []
    for exp_dir in exp_dirs:
        exp_path = os.path.join(scan_dir, exp_dir)
        plots_dir = os.path.join(exp_path, "plots")
        if os.path.exists(plots_dir):
            mtime = os.path.getmtime(plots_dir)
            exp_dirs_with_time.append((mtime, exp_dir, exp_path))
    
    if not exp_dirs_with_time:
        print(f"[ERROR] No plots directories found in {scan_dir}")
        return None
    
    # Get the latest experiment
    exp_dirs_with_time.sort(reverse=True)
    latest_exp_path = exp_dirs_with_time[0][2]
    
    # Find mesh files in plots directory
    plots_dir = os.path.join(latest_exp_path, "plots")
    mesh_files = glob.glob(os.path.join(plots_dir, "mesh_*.ply"))
    
    if not mesh_files:
        print(f"[ERROR] No mesh files found in {plots_dir}")
        return None
    
    # Sort by epoch number (mesh_2400.ply > mesh_1200.ply)
    def get_epoch(mesh_file):
        basename = os.path.basename(mesh_file)
        try:
            # Extract epoch number from mesh_2400.ply -> 2400
            epoch = int(basename.replace("mesh_", "").replace(".ply", ""))
            return epoch
        except:
            return 0
    
    mesh_files.sort(key=get_epoch, reverse=True)
    latest_mesh = mesh_files[0]
    
    print(f"[INFO] Scan {scan_id}: Found mesh {os.path.basename(latest_mesh)} in {latest_exp_path}")
    return latest_mesh


def evaluate_scan(scan_id, mesh_file, data_dir, output_dir):
    """Evaluate a single scan using evaluate_single_scene.py"""
    script_path = os.path.join(os.path.dirname(__file__), "evals", "replica_eval", "evaluate_single_scene.py")
    
    if not os.path.exists(script_path):
        print(f"[ERROR] Evaluation script not found: {script_path}")
        return False
    
    cmd = [
        sys.executable,
        script_path,
        "--input_mesh", mesh_file,
        "--scan_id", str(scan_id),
        "--data_dir", data_dir,
        "--output_dir", output_dir
    ]
    
    print(f"\n{'='*60}")
    print(f"Evaluating Scan {scan_id} (room{scan_id-1})")
    print(f"{'='*60}")
    print(f"Mesh: {mesh_file}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"[SUCCESS] Scan {scan_id} evaluation completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Scan {scan_id} evaluation failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"[ERROR] Scan {scan_id} evaluation failed: {e}")
        return False


def main():
    """Main function to evaluate scans 1, 2, and 3 from runs_unc_beta"""
    print("="*60)
    print("EVALUATING SCANS 1, 2, 3 FROM runs_unc_beta")
    print("(First iteration - likely no max clamp)")
    print("="*60)
    
    # Configuration
    runs_dir = "runs_unc_beta"
    data_dir = "data/Replica"
    base_output_dir = "evaluation_results_unc_beta"
    
    # Check if runs directory exists
    if not os.path.exists(runs_dir):
        print(f"[ERROR] Runs directory not found: {runs_dir}")
        print("Please run this script from the ND-Uncertainty project root")
        sys.exit(1)
    
    # Evaluate each scan
    results = {}
    for scan_id in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"Processing Scan {scan_id}")
        print(f"{'='*60}")
        
        # Find latest mesh
        mesh_file = find_latest_mesh(runs_dir, scan_id)
        if mesh_file is None:
            print(f"[SKIP] Scan {scan_id}: Could not find mesh file")
            results[scan_id] = False
            continue
        
        # Evaluate
        output_dir = os.path.join(base_output_dir, f"scan{scan_id}")
        success = evaluate_scan(scan_id, mesh_file, data_dir, output_dir)
        results[scan_id] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for scan_id in [1, 2, 3]:
        status = "[SUCCESS]" if results.get(scan_id, False) else "[FAILED]"
        print(f"Scan {scan_id}: {status}")
    
    # Check results files
    print(f"\n{'='*60}")
    print("RESULTS FILES")
    print("="*60)
    
    for scan_id in [1, 2, 3]:
        results_file = os.path.join(base_output_dir, f"scan{scan_id}", "evaluation_results.txt")
        if os.path.exists(results_file):
            print(f"\nScan {scan_id} results:")
            with open(results_file, 'r') as f:
                print(f.read())
        else:
            print(f"Scan {scan_id}: Results file not found at {results_file}")
    
    print(f"\n{'='*60}")
    print("Done! Check evaluation_results_unc_beta/scan*/evaluation_results.txt for detailed metrics")
    print("="*60)


if __name__ == "__main__":
    main()
