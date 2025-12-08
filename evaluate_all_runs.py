#!/usr/bin/env python3
"""
Evaluation script for all replica runs in runs_unc_beta.
Finds the most recent replica trained results for each scan and formats them in a table.
"""

import argparse
import os
import glob
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import re

# Try to import eval_recon functions directly, otherwise use subprocess
USE_DIRECT_IMPORT = False
eval_recon_paths = [
    os.path.join(os.path.dirname(__file__), '..', 'ND-SDF', 'evals', 'replica_eval', 'eval_recon.py'),
    os.path.join(os.path.dirname(__file__), 'evals', 'replica_eval', 'eval_recon.py'),
]

for eval_path in eval_recon_paths:
    if os.path.exists(eval_path):
        try:
            sys.path.insert(0, os.path.dirname(eval_path))
            from eval_recon import calc_3d_metric  # type: ignore
            USE_DIRECT_IMPORT = True
            break
        except ImportError:
            continue

if not USE_DIRECT_IMPORT:
    print("Warning: Could not import eval_recon directly, will use subprocess instead")


def find_latest_mesh(plots_dir):
    """Find the latest mesh file in the plots directory."""
    if not os.path.exists(plots_dir):
        return None
    
    # Look for .ply files
    mesh_files = glob.glob(os.path.join(plots_dir, "*.ply"))
    if not mesh_files:
        return None
    
    # Sort by modification time, most recent first
    mesh_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return mesh_files[0]


def find_most_recent_experiment(runs_dir, exp_name, scan_idx):
    """Find the most recent experiment matching the pattern.
    
    Looks for directories that start with exp_name and end with _{scan_idx}.
    Handles both 'replica_1' and 'replica_all8_1' patterns.
    Also handles both structures: with timestamp subdirectories and without.
    """
    if not os.path.exists(runs_dir):
        return None
    
    # Find all experiments matching the pattern: exp_name*_{scan_idx}
    # This handles both 'replica_1' and 'replica_all8_1'
    pattern = f"^{re.escape(exp_name)}.*_{scan_idx}$"
    matching_exps = []
    
    for item in os.listdir(runs_dir):
        if re.match(pattern, item):
            exp_path = os.path.join(runs_dir, item)
            if os.path.isdir(exp_path):
                # Check if there are timestamp subdirectories
                timestamp_dirs = []
                has_timestamp_dirs = False
                
                for subdir in os.listdir(exp_path):
                    subdir_path = os.path.join(exp_path, subdir)
                    if os.path.isdir(subdir_path):
                        # Check if it's a timestamp directory (YYYY-MM-DD_HH-MM-SS format)
                        if re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', subdir):
                            timestamp_dirs.append((subdir_path, subdir))
                            has_timestamp_dirs = True
                
                if has_timestamp_dirs:
                    # Structure with timestamp subdirectories
                    if timestamp_dirs:
                        # Sort by timestamp, most recent first
                        timestamp_dirs.sort(key=lambda x: x[1], reverse=True)
                        matching_exps.append((timestamp_dirs[0][0], timestamp_dirs[0][1], 'timestamp'))
                else:
                    # Structure without timestamp subdirectories (flat structure)
                    # Use the experiment directory itself, sorted by modification time
                    matching_exps.append((exp_path, os.path.getmtime(exp_path), 'mtime'))
    
    if not matching_exps:
        return None
    
    # Return the most recent one
    # If using timestamps, sort by timestamp string; if using mtime, sort by mtime
    if matching_exps[0][2] == 'timestamp':
        # Timestamp-based sorting
        matching_exps.sort(key=lambda x: x[1], reverse=True)
    else:
        # Modification time-based sorting
        matching_exps.sort(key=lambda x: x[1], reverse=True)
    
    return matching_exps[0][0]


def parse_eval_output(output_text):
    """Parse the evaluation output to extract metrics."""
    # The output format is:
    # Acc Comp Chamfer Prec Recal F-score Normal Acc Normal Comp Normal Avg
    # <values>
    lines = output_text.strip().split('\n')
    if len(lines) < 2:
        return None
    
    # Find the line with values
    values_line = None
    for line in lines:
        if line.strip() and not line.startswith('Acc Comp'):
            # Try to parse as numbers
            try:
                values = [float(x) for x in line.split()]
                if len(values) >= 9:
                    values_line = values
                    break
            except ValueError:
                continue
    
    if values_line is None:
        return None
    
    # Extract metrics: Acc, Comp, Chamfer, Prec, Recal, F-score, Normal Acc, Normal Comp, Normal Avg
    return {
        'accuracy': values_line[0],
        'completion': values_line[1],
        'chamfer': values_line[2],
        'precision': values_line[3],
        'recall': values_line[4],
        'fscore': values_line[5],
        'normal_acc': values_line[6],
        'normal_comp': values_line[7],
        'normal_avg': values_line[8]
    }


def fix_mesh_if_scene(mesh_file):
    """If mesh file is a Scene (multiple meshes), convert to single mesh and save."""
    try:
        import trimesh
        mesh = trimesh.load(mesh_file, process=False)
        
        # Check if it's a Scene (multiple meshes)
        if isinstance(mesh, trimesh.Scene):
            print(f"  Converting Scene (multiple meshes) to single mesh...")
            # Combine all meshes in the scene into one
            combined_mesh = trimesh.util.concatenate([mesh.geometry[key] for key in mesh.geometry.keys()])
            # Save the combined mesh to a temporary file
            temp_file = mesh_file.replace('.ply', '_combined.ply')
            combined_mesh.export(temp_file)
            print(f"  Created combined mesh: {temp_file}")
            return temp_file
        return mesh_file
    except Exception as e:
        print(f"  Warning: Could not fix Scene mesh: {e}, using original")
        return mesh_file


def evaluate_mesh(rec_mesh, gt_mesh, cull_mesh_script=None, data_dir=None, scan_idx=None):
    """Evaluate a mesh file and return metrics."""
    # Fix Scene objects before evaluation
    rec_mesh = fix_mesh_if_scene(rec_mesh)
    
    if USE_DIRECT_IMPORT:
        # Use direct import
        try:
            # We need to call calc_3d_metric, but it prints to stdout
            # So we'll capture stdout
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                calc_3d_metric(rec_mesh, gt_mesh, align=False)
            output = f.getvalue()
            return parse_eval_output(output)
        except Exception as e:
            print(f"Error in direct evaluation: {e}")
            return None
    else:
        # Use subprocess - try multiple possible locations
        eval_script = None
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'ND-SDF', 'evals', 'replica_eval', 'eval_recon.py'),
            os.path.join(os.path.dirname(__file__), 'evals', 'replica_eval', 'eval_recon.py'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                eval_script = path
                break
        
        if eval_script is None:
            print(f"Error: eval_recon.py not found. Tried: {possible_paths}")
            return None
        
        # rec_mesh is already fixed by fix_mesh_if_scene at the start of evaluate_mesh
        cmd = f"python {eval_script} --rec_mesh {rec_mesh} --gt_mesh {gt_mesh}"
        try:
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8")
            return parse_eval_output(output)
        except subprocess.CalledProcessError as e:
            print(f"Error running evaluation: {e}")
            error_output = e.output.decode('utf-8') if e.output else 'No output'
            print(f"Output: {error_output[:500]}")  # Show first 500 chars
            return None


def cull_mesh_if_needed(mesh_file, data_dir, scan_idx, scan_name, out_dir):
    """Cull mesh if cull_mesh.py script is available."""
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'ND-SDF', 'evals', 'replica_eval', 'cull_mesh.py'),
        os.path.join(os.path.dirname(__file__), 'evals', 'replica_eval', 'cull_mesh.py'),
        os.path.join(os.path.dirname(__file__), 'scripts', 'cull_mesh.py'),
    ]
    
    cull_script = None
    for path in possible_paths:
        if os.path.exists(path):
            cull_script = path
            break
    
    if cull_script and data_dir and scan_idx:
        cull_mesh_out = os.path.join(out_dir, f"cull_{scan_name}.ply")
        cameras_file = os.path.join(data_dir, f"scan{scan_idx}", "cameras.npz")
        traj_file = os.path.join(data_dir, f"scan{scan_idx}", "traj.txt")
        
        if os.path.exists(cameras_file) and os.path.exists(traj_file):
            cmd = f"python {cull_script} --input_mesh {mesh_file} --input_scalemat {cameras_file} --traj {traj_file} --output_mesh {cull_mesh_out}"
            try:
                result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                if os.path.exists(cull_mesh_out):
                    return cull_mesh_out
                else:
                    print(f"  Warning: Cull mesh command succeeded but output file not found")
                    return mesh_file
            except subprocess.CalledProcessError as e:
                print(f"  Warning: Failed to cull mesh (exit code {e.returncode})")
                if e.stderr:
                    print(f"  Cull error: {e.stderr[:200]}")  # Show first 200 chars of error
                return mesh_file
    
    return mesh_file


def main():
    parser = argparse.ArgumentParser(description='Evaluate all replica runs and format results in a table')
    parser.add_argument('--runs_dir', type=str, default='runs_unc_beta', help='Root directory containing runs')
    parser.add_argument('--data_dir', type=str, default='./data/Replica', help='Path to Replica dataset')
    parser.add_argument('--exp_name', type=str, default='replica', help='Experiment name pattern')
    parser.add_argument('--out_dir', type=str, default='evaluation_results', help='Output directory for culled meshes')
    parser.add_argument('--skip_cull', action='store_true', help='Skip mesh culling step')
    args = parser.parse_args()
    
    scans = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    results = {}
    
    print("Finding most recent experiments for each scan...")
    for idx, scan in enumerate(scans):
        scan_idx = idx + 1
        
        print(f"\nProcessing {scan} (scan{scan_idx})...")
        
        # Find most recent experiment (handles both 'replica_1' and 'replica_all8_1' patterns)
        exp_dir = find_most_recent_experiment(args.runs_dir, args.exp_name, scan_idx)
        if exp_dir is None:
            print(f"  No experiment found for {scan}")
            results[scan] = None
            continue
        
        print(f"  Found experiment: {exp_dir}")
        
        # Find latest mesh
        plots_dir = os.path.join(exp_dir, 'plots')
        mesh_file = find_latest_mesh(plots_dir)
        if mesh_file is None:
            print(f"  No mesh file found in {plots_dir}")
            results[scan] = None
            continue
        
        print(f"  Using mesh: {mesh_file}")
        
        # Cull mesh if needed
        if not args.skip_cull:
            culled_mesh = cull_mesh_if_needed(mesh_file, args.data_dir, scan_idx, scan, args.out_dir)
        else:
            culled_mesh = mesh_file
        
        # Find GT mesh
        gt_mesh = os.path.join(args.data_dir, 'cull_GTmesh', f"{scan}.ply")
        if not os.path.exists(gt_mesh):
            # Try alternative location
            gt_mesh = os.path.join(args.data_dir, 'cull_GTmesh', f"{scan}.ply")
            if not os.path.exists(gt_mesh):
                print(f"  Warning: GT mesh not found at {gt_mesh}")
                results[scan] = None
                continue
        
        # Evaluate
        print(f"  Evaluating against GT: {gt_mesh}")
        metrics = evaluate_mesh(culled_mesh, gt_mesh)
        
        if metrics is None:
            print(f"  Failed to evaluate {scan}")
            results[scan] = None
        else:
            results[scan] = metrics
            print(f"  Results: Normal C.={metrics['normal_avg']:.2f}, Chamfer={metrics['chamfer']:.2f}, F-score={metrics['fscore']:.2f}")
    
    # Print table
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"\n{'Scan':<12} {'ND-Uncertainty':<20}")
    print(f"{'':12} {'Normal C. ↑':<15} {'Chamfer ↓':<15} {'F-score ↑':<15}")
    print("-" * 80)
    
    valid_results = []
    for scan in scans:
        if results[scan] is None:
            print(f"{scan:<12} {'':<15} {'':<15} {'':<15}")
        else:
            m = results[scan]
            print(f"{scan:<12} {m['normal_avg']:<15.2f} {m['chamfer']:<15.2f} {m['fscore']:<15.2f}")
            valid_results.append(m)
    
    print("-" * 80)
    
    # Calculate averages
    if valid_results:
        avg_normal = sum(m['normal_avg'] for m in valid_results) / len(valid_results)
        avg_chamfer = sum(m['chamfer'] for m in valid_results) / len(valid_results)
        avg_fscore = sum(m['fscore'] for m in valid_results) / len(valid_results)
        print(f"{'Average':<12} {avg_normal:<15.2f} {avg_chamfer:<15.2f} {avg_fscore:<15.2f}")
    
    print("="*80)
    
    # Save results to file
    results_file = os.path.join(args.out_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"\n{'Scan':<12} {'ND-Uncertainty':<20}\n")
        f.write(f"{'':12} {'Normal C. ↑':<15} {'Chamfer ↓':<15} {'F-score ↑':<15}\n")
        f.write("-" * 80 + "\n")
        
        for scan in scans:
            if results[scan] is None:
                f.write(f"{scan:<12} {'':<15} {'':<15} {'':<15}\n")
            else:
                m = results[scan]
                f.write(f"{scan:<12} {m['normal_avg']:<15.2f} {m['chamfer']:<15.2f} {m['fscore']:<15.2f}\n")
        
        f.write("-" * 80 + "\n")
        
        if valid_results:
            avg_normal = sum(m['normal_avg'] for m in valid_results) / len(valid_results)
            avg_chamfer = sum(m['chamfer'] for m in valid_results) / len(valid_results)
            avg_fscore = sum(m['fscore'] for m in valid_results) / len(valid_results)
            f.write(f"{'Average':<12} {avg_normal:<15.2f} {avg_chamfer:<15.2f} {avg_fscore:<15.2f}\n")
        
        f.write("="*80 + "\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
