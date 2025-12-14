#!/usr/bin/env python3
"""
Extract and plot loss curves from TensorBoard logs for ND-Uncertainty training.
Specifically designed to compare runs_unc_w0_1 and runs_unc_w0_5.

Usage:
    python plot_losses_w0_comparison.py
"""

import os
import glob
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Try different import methods for TensorBoard
try:
    from tensorflow.python.summary import summary_iterator
    import tensorflow as tf
    HAS_TF = True
    HAS_TB = False
except ImportError:
    try:
        from tensorboard.backend.event_processing import event_accumulator
        HAS_TB = True
        HAS_TF = False
    except ImportError:
        print("Error: Need either tensorflow or tensorboard installed")
        print("Install with: conda install tensorflow or pip install tensorboard")
        raise

def extract_metric_from_tensorboard(log_dir, metric_tag):
    """Extract a metric from TensorBoard event files."""
    event_files = sorted(glob.glob(os.path.join(log_dir, 'events.out.tfevents.*')))
    if not event_files:
        print(f"Warning: No event files found in {log_dir}")
        return None
    
    # Use the most recent event file
    event_file = event_files[-1]
    print(f"Reading from: {event_file}")
    
    steps = []
    values = []
    
    try:
        if HAS_TF:
            # Use tensorflow's summary_iterator (reads tensor values correctly)
            for event in summary_iterator.summary_iterator(event_file):
                if event.summary:
                    for value in event.summary.value:
                        if value.tag == metric_tag:
                            step = event.step
                            # Try to get value from tensor (not simple_value)
                            val = None
                            if hasattr(value, 'tensor') and value.tensor:
                                try:
                                    arr = tf.make_ndarray(value.tensor)
                                    if arr.size == 1:
                                        val = float(arr.item())
                                except:
                                    pass
                            
                            # Fallback to simple_value if tensor is not available
                            if val is None and hasattr(value, 'simple_value'):
                                val = value.simple_value
                            
                            if val is not None:
                                steps.append(step)
                                values.append(val)
        else:
            # Use tensorboard's event_accumulator (fallback)
            ea = event_accumulator.EventAccumulator(log_dir, size_guidance={
                event_accumulator.SCALARS: 0,
            })
            ea.Reload()
            
            if metric_tag in ea.Tags().get('scalars', []):
                scalar_events = ea.Scalars(metric_tag)
                for event in scalar_events:
                    steps.append(event.step)
                    values.append(event.value)
    except Exception as e:
        print(f"Error reading {metric_tag}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    if not steps:
        print(f"Warning: No data found for {metric_tag}")
        return None
    
    return {'steps': steps, 'values': values}

def find_latest_run_in_dir(runs_dir, pattern):
    """Find the latest run directory matching a pattern."""
    matching_dirs = []
    for item in os.listdir(runs_dir):
        if os.path.isdir(os.path.join(runs_dir, item)) and pattern in item:
            matching_dirs.append(item)
    
    if not matching_dirs:
        return None
    
    # Find the most recent run within each matching directory
    latest_run = None
    latest_time = None
    
    for dir_name in matching_dirs:
        dir_path = os.path.join(runs_dir, dir_name)
        # Look for timestamped subdirectories (YYYY-MM-DD_HH-MM-SS)
        for subdir in os.listdir(dir_path):
            subdir_path = os.path.join(dir_path, subdir)
            if os.path.isdir(subdir_path) and len(subdir) == 19 and subdir[4] == '-':
                # Check if logs directory exists
                logs_path = os.path.join(subdir_path, 'logs')
                if os.path.isdir(logs_path):
                    # Use directory name as timestamp (YYYY-MM-DD_HH-MM-SS)
                    if latest_time is None or subdir > latest_time:
                        latest_time = subdir
                        latest_run = logs_path
    
    return latest_run

def process_weight_run(runs_dir, weight_config, lam_value, base_dir):
    """Process a single weight configuration run."""
    print(f"\n{'='*60}")
    print(f"Processing: weight={weight_config}, lambda={lam_value}")
    print(f"{'='*60}")
    
    # Pattern: replica_weight{weight_config}_lam{lam_value}_1
    pattern = f"replica_weight{weight_config}_lam{lam_value}"
    log_dir = find_latest_run_in_dir(runs_dir, pattern)
    
    if log_dir is None:
        print(f"Error: No run found for {pattern} in {runs_dir}")
        return None
    
    print(f"Using log directory: {log_dir}")
    
    # Loss tags to extract
    loss_tags = {
        'total': 'loss/total',
        'eik': 'loss/eik',
        'rgb_l1': 'loss/rgb_l1',
        'l_ssim': 'loss/l_ssim',
        'uncertainty_loss': 'loss/uncertainty_loss',
        'variance_regularizer': 'loss/variance_regularizer',
        'smooth': 'loss/smooth',
        'curvature': 'loss/curvature',
        'ab_normal_l1': 'loss/ab_normal_l1',
        'ab_normal_cos': 'loss/ab_normal_cos',
        'ab_biased_l1': 'loss/ab_biased_l1',
        'ab_biased_cos': 'loss/ab_biased_cos',
        'ab_depth': 'loss/ab_depth',
    }
    
    # Extract all losses
    loss_data = {}
    for loss_name, loss_tag in loss_tags.items():
        print(f"Extracting {loss_name} ({loss_tag})...")
        data = extract_metric_from_tensorboard(log_dir, loss_tag)
        loss_data[loss_name] = data
    
    return loss_data

def plot_comparison(metric_name, w0_1_data, w0_5_data, lam_value, output_path):
    """Plot comparison between w0_1 and w0_5 for a specific metric."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot w0_1 (weight=0.1)
    if w0_1_data is not None:
        ax.plot(w0_1_data['steps'], w0_1_data['values'], 
               color='blue', linewidth=1.5, 
               label=f'weight=0.1', alpha=0.8, linestyle='-')
    
    # Plot w0_5 (weight=0.5)
    if w0_5_data is not None:
        ax.plot(w0_5_data['steps'], w0_5_data['values'], 
               color='red', linewidth=1.5, 
               label=f'weight=0.5', alpha=0.8, linestyle='-')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Comparison: {metric_name.replace("_", " ").title()} (lambda={lam_value})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def plot_all_metrics_comparison(w0_1_data_dict, w0_5_data_dict, lam_value, output_dir):
    """Plot all metrics in a grid for comparison."""
    metrics_to_plot = ['total', 'uncertainty_loss', 'rgb_l1', 'l_ssim', 
                       'variance_regularizer', 'curvature', 'ab_normal_l1']
    
    n_metrics = len(metrics_to_plot)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        w0_1_data = w0_1_data_dict.get(metric_name)
        w0_5_data = w0_5_data_dict.get(metric_name)
        
        if w0_1_data is not None:
            ax.plot(w0_1_data['steps'], w0_1_data['values'], 
                   color='blue', linewidth=1.5, 
                   label='weight=0.1', alpha=0.8, linestyle='-')
        
        if w0_5_data is not None:
            ax.plot(w0_5_data['steps'], w0_5_data['values'], 
                   color='red', linewidth=1.5, 
                   label='weight=0.5', alpha=0.8, linestyle='-')
        
        ax.set_xlabel('Training Step', fontsize=10)
        ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=10)
        ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Weight Comparison (lambda={lam_value}): weight=0.1 (blue) vs weight=0.5 (red)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'all_metrics_comparison_lam{lam_value}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot loss curves comparing runs_unc_w0_1 and runs_unc_w0_5')
    parser.add_argument('--base-dir', type=str,
                       default='/home/hussein/project/ND-Uncertainty',
                       help='Base directory (default: /home/hussein/project/ND-Uncertainty)')
    parser.add_argument('--lam-values', nargs='+', type=str,
                       default=['01', '05'],
                       help='Lambda values to process (default: 01 05)')
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    runs_w0_1_dir = os.path.join(base_dir, 'runs_unc_w0_1')
    runs_w0_5_dir = os.path.join(base_dir, 'runs_unc_w0_5')
    
    if not os.path.exists(runs_w0_1_dir):
        print(f"Error: Directory not found: {runs_w0_1_dir}")
        return
    
    if not os.path.exists(runs_w0_5_dir):
        print(f"Error: Directory not found: {runs_w0_5_dir}")
        return
    
    print("ND-Uncertainty Weight Comparison Plotter")
    print("="*60)
    print(f"Comparing: runs_unc_w0_1 (weight=0.1) vs runs_unc_w0_5 (weight=0.5)")
    print("="*60)
    
    output_base_dir = os.path.join(base_dir, 'all_losses', 'weight_comparison')
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Process each lambda value
    for lam_value in args.lam_values:
        print(f"\n{'='*60}")
        print(f"Processing lambda={lam_value}")
        print(f"{'='*60}")
        
        # Process w0_1
        w0_1_data = process_weight_run(runs_w0_1_dir, '01', lam_value, base_dir)
        
        # Process w0_5
        w0_5_data = process_weight_run(runs_w0_5_dir, '05', lam_value, base_dir)
        
        if w0_1_data is None and w0_5_data is None:
            print(f"Skipping lambda={lam_value} - no data found")
            continue
        
        # Create output directory for this lambda
        output_dir = os.path.join(output_base_dir, f'lam_{lam_value}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot individual metric comparisons
        metrics_to_plot = ['total', 'uncertainty_loss', 'rgb_l1', 'l_ssim', 
                          'variance_regularizer', 'curvature', 'ab_normal_l1',
                          'ab_normal_cos', 'ab_biased_l1', 'ab_depth']
        
        for metric_name in metrics_to_plot:
            w0_1_metric = w0_1_data.get(metric_name) if w0_1_data else None
            w0_5_metric = w0_5_data.get(metric_name) if w0_5_data else None
            
            if w0_1_metric is not None or w0_5_metric is not None:
                output_path = os.path.join(output_dir, f'{metric_name}_comparison.png')
                plot_comparison(metric_name, w0_1_metric, w0_5_metric, lam_value, output_path)
        
        # Plot all metrics in one grid
        if w0_1_data or w0_5_data:
            plot_all_metrics_comparison(w0_1_data or {}, w0_5_data or {}, lam_value, output_dir)
        
        # Save raw data as JSON
        json_data = {
            'w0_1': {},
            'w0_5': {}
        }
        
        if w0_1_data:
            for metric_name, data in w0_1_data.items():
                if data is not None:
                    json_data['w0_1'][metric_name] = {
                        'steps': data['steps'],
                        'values': data['values']
                    }
        
        if w0_5_data:
            for metric_name, data in w0_5_data.items():
                if data is not None:
                    json_data['w0_5'][metric_name] = {
                        'steps': data['steps'],
                        'values': data['values']
                    }
        
        json_path = os.path.join(output_dir, 'comparison_data.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"\nSaved raw data to: {json_path}")
    
    print(f"\n{'='*60}")
    print("Done! Check the all_losses/weight_comparison/ directory for plots.")
    print(f"  - Individual comparisons: all_losses/weight_comparison/lam_*/")
    print(f"  - Grid plots: all_losses/weight_comparison/lam_*/all_metrics_comparison_lam*.png")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

