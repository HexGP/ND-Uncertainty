#!/usr/bin/env python3
"""
Extract and plot loss curves from TensorBoard logs for ND-SDF training.

Usage:
    python plot_losses.py
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

def plot_metric(data, title, ylabel, output_path, color='blue'):
    """Plot a single metric."""
    if data is None:
        print(f"Skipping {title} - no data")
        return False
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['steps'], data['values'], color=color, linewidth=1.5)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return True

def plot_combined_base_loss(loss_data_dict, output_path):
    """Plot all individual losses together with the total loss."""
    if not loss_data_dict:
        print("Skipping combined plot - no data")
        return False
    
    # Color palette for different losses
    colors = plt.cm.tab10(np.linspace(0, 1, len(loss_data_dict)))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot total loss first (thicker line, different style)
    if 'total' in loss_data_dict and loss_data_dict['total'] is not None:
        total_data = loss_data_dict['total']
        ax.plot(total_data['steps'], total_data['values'], 
               color='black', linewidth=2.5, label='Total Loss', linestyle='-', alpha=0.9)
    
    # Plot individual losses
    color_idx = 0
    for loss_name, data in loss_data_dict.items():
        if loss_name == 'total' or data is None:
            continue
        
        # Skip if we've used too many colors
        if color_idx >= len(colors):
            color_idx = 0
        
        ax.plot(data['steps'], data['values'], 
               color=colors[color_idx], linewidth=1.5, 
               label=loss_name.replace('_', ' ').title(), alpha=0.8)
        color_idx += 1
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('ND-Uncertainty: Combined Base Loss Curves', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return True

def plot_combined_all_scans_total_loss(all_scans_data, output_path):
    """Plot total loss from all 8 scans in one image with different colors."""
    if not all_scans_data:
        print("Skipping combined all-scans plot - no data")
        return False
    
    # Specific color palette for 8 scans: red, green, blue, orange, purple, lime, pink, teal
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'lime', 'pink', 'teal']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot each scan's total loss
    for scan_id, data in sorted(all_scans_data.items()):
        if data is None:
            continue
        
        color_idx = scan_id - 1  # scan_id is 1-8, color_idx should be 0-7
        if color_idx < len(colors):
            ax.plot(data['steps'], data['values'], 
                   color=colors[color_idx], linewidth=1.0, 
                   label=f'Scan {scan_id}', alpha=0.8)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.set_title('ND-Uncertainty: Total Loss - All 8 Scans', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return True

def plot_combined_all_scans_uncertainty_loss(all_scans_data, output_path):
    """Plot uncertainty loss from all 8 scans in one image with different colors."""
    if not all_scans_data:
        print("Skipping combined uncertainty loss plot - no data")
        return False
    
    # Specific color palette for 8 scans: red, green, blue, orange, purple, lime, pink, teal
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'lime', 'pink', 'teal']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot each scan's uncertainty loss
    for scan_id, data in sorted(all_scans_data.items()):
        if data is None:
            continue
        
        color_idx = scan_id - 1  # scan_id is 1-8, color_idx should be 0-7
        if color_idx < len(colors):
            ax.plot(data['steps'], data['values'], 
                   color=colors[color_idx], linewidth=1.0, 
                   label=f'Scan {scan_id}', alpha=0.8)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Uncertainty Loss', fontsize=12)
    ax.set_title('ND-Uncertainty: Uncertainty Loss - All 8 Scans', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return True

def plot_combined_total_and_uncertainty_loss_single_scan(total_loss_data, uncertainty_loss_data, scan_id, output_path):
    """Plot both total loss (blue) and uncertainty loss (red) for a single scan."""
    if total_loss_data is None and uncertainty_loss_data is None:
        print(f"Skipping combined plot for scan {scan_id} - no data")
        return False
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot total loss (blue)
    if total_loss_data is not None:
        ax.plot(total_loss_data['steps'], total_loss_data['values'], 
               color='blue', linewidth=1.0, 
               label='Total Loss', alpha=0.8, linestyle='-')
    
    # Plot uncertainty loss (red)
    if uncertainty_loss_data is not None:
        ax.plot(uncertainty_loss_data['steps'], uncertainty_loss_data['values'], 
               color='red', linewidth=1.0, 
               label='Uncertainty Loss', alpha=0.8, linestyle='-')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'ND-Uncertainty Scan {scan_id}: Total Loss (Blue) & Uncertainty Loss (Red)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return True

def plot_combined_total_and_uncertainty_loss(all_scans_total_loss, all_scans_uncertainty_loss, output_path):
    """Plot both total loss (blue) and uncertainty loss (red) for all 8 scans."""
    if not all_scans_total_loss and not all_scans_uncertainty_loss:
        print("Skipping combined total+uncertainty plot - no data")
        return False
    
    # Specific color palette for 8 scans: red, green, blue, orange, purple, lime, pink, teal
    scan_colors = ['red', 'green', 'blue', 'orange', 'purple', 'lime', 'pink', 'teal']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot total loss for all scans (in blue)
    for scan_id, data in sorted(all_scans_total_loss.items()):
        if data is None:
            continue
        ax.plot(data['steps'], data['values'], 
               color='blue', linewidth=1.0, 
               label=f'Total Loss - Scan {scan_id}', alpha=0.7, linestyle='-')
    
    # Plot uncertainty loss for all scans (in red)
    for scan_id, data in sorted(all_scans_uncertainty_loss.items()):
        if data is None:
            continue
        ax.plot(data['steps'], data['values'], 
               color='red', linewidth=1.0, 
               label=f'Uncertainty Loss - Scan {scan_id}', alpha=0.7, linestyle='--')
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('ND-Uncertainty: Total Loss (Blue) & Uncertainty Loss (Red) - All 8 Scans', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return True

def find_latest_run(runs_dir, scan_id=None):
    """Find the latest run directory."""
    if scan_id:
        # Look for replica_all8_{scan_id} pattern (ND-Uncertainty) or replica_{scan_id} (ND-SDF)
        patterns = [f"replica_all8_{scan_id}", f"replica_{scan_id}"]
    else:
        # Look for any replica_* pattern
        patterns = ["replica_all8_*", "replica_*"]
    
    matching_dirs = []
    for item in os.listdir(runs_dir):
        if os.path.isdir(os.path.join(runs_dir, item)):
            for pattern in patterns:
                if pattern.replace('*', '') in item or (scan_id and f"_{scan_id}" in item):
                    matching_dirs.append(item)
                    break
    
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

def process_replica_run(runs_dir, scan_id, base_dir):
    """Process a single replica run."""
    print(f"\n{'='*60}")
    print(f"Processing: Replica scan {scan_id}")
    print(f"{'='*60}")
    
    # Find the latest run
    log_dir = find_latest_run(runs_dir, scan_id)
    
    if log_dir is None:
        print(f"Error: No run found for scan {scan_id} in {runs_dir}")
        return None
    
    print(f"Using log directory: {log_dir}")
    
    # Output directory per scan
    output_dir = os.path.join(base_dir, 'all_losses', 'Replica', f'scan_{scan_id}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Common loss components in ND-SDF + ND-Uncertainty
    loss_tags = {
        'total': 'loss/total',
        'eik': 'loss/eik',
        'rgb_l1': 'loss/rgb_l1',
        'rgb_mse': 'loss/rgb_mse',
        'smooth': 'loss/smooth',
        'curvature': 'loss/curvature',
        'normal_l1': 'loss/normal_l1',
        'normal_cos': 'loss/normal_cos',
        'depth': 'loss/depth',
        'ab_normal_l1': 'loss/ab_normal_l1',
        'ab_normal_cos': 'loss/ab_normal_cos',
        'ab_biased_l1': 'loss/ab_biased_l1',
        'ab_biased_cos': 'loss/ab_biased_cos',
        'ab_depth': 'loss/ab_depth',
        # ND-Uncertainty specific losses
        'uncertainty_loss': 'loss/uncertainty_loss',
        'l_ssim': 'loss/l_ssim',
        'variance_regularizer': 'loss/variance_regularizer',
    }
    
    # Extract all losses
    loss_data = {}
    for loss_name, loss_tag in loss_tags.items():
        print(f"\nExtracting {loss_name} ({loss_tag})...")
        data = extract_metric_from_tensorboard(log_dir, loss_tag)
        loss_data[loss_name] = data
    
    # Color mapping for individual plots
    color_map = {
        'total': 'black',
        'eik': 'blue',
        'rgb_l1': 'red',
        'rgb_mse': 'orange',
        'smooth': 'green',
        'curvature': 'purple',
        'normal_l1': 'brown',
        'normal_cos': 'pink',
        'depth': 'cyan',
        'ab_normal_l1': 'olive',
        'ab_normal_cos': 'navy',
        'ab_biased_l1': 'teal',
        'ab_biased_cos': 'maroon',
        'ab_depth': 'gold',
        # ND-Uncertainty specific
        'uncertainty_loss': 'crimson',
        'l_ssim': 'darkorange',
        'variance_regularizer': 'mediumvioletred',
    }
    
    # Plot individual losses
    for loss_name, data in loss_data.items():
        if data is not None:
            # Format title - avoid double "Loss" (e.g., "Uncertainty Loss Loss")
            loss_display = loss_name.replace('_', ' ').title()
            if 'loss' in loss_name.lower():
                title = f"ND-Uncertainty Scan {scan_id}: {loss_display}"
            else:
                title = f"ND-Uncertainty Scan {scan_id}: {loss_display} Loss"
            ylabel = "Loss"
            color = color_map.get(loss_name, 'blue')
            # Avoid double _loss suffix (e.g., uncertainty_loss_loss.png)
            if loss_name.endswith('_loss'):
                filename = f'{loss_name}.png'
            else:
                filename = f'{loss_name}_loss.png'
            output_path = os.path.join(output_dir, filename)
            plot_metric(data, title, ylabel, output_path, color)
    
    # Plot combined base loss for this scan
    output_path = os.path.join(output_dir, 'combined_base_loss.png')
    plot_combined_base_loss(loss_data, output_path)
    
    # Plot combined total + uncertainty loss for this scan
    total_loss_data = loss_data.get('total', None)
    uncertainty_loss_data = loss_data.get('uncertainty_loss', None)
    if total_loss_data is not None or uncertainty_loss_data is not None:
        output_path = os.path.join(output_dir, 'combined_total_unc_loss.png')
        plot_combined_total_and_uncertainty_loss_single_scan(
            total_loss_data, uncertainty_loss_data, scan_id, output_path)
    
    # Save raw data as JSON
    json_data = {}
    for loss_name, data in loss_data.items():
        if data is not None:
            json_data[loss_name] = {
                'steps': data['steps'],
                'values': data['values']
            }
    
    json_path = os.path.join(output_dir, 'loss_data.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nSaved raw data to: {json_path}")
    
    # Return the total loss and uncertainty loss data for combined plotting
    return {
        'total': loss_data.get('total', None),
        'uncertainty_loss': loss_data.get('uncertainty_loss', None)
    }

def main():
    parser = argparse.ArgumentParser(description='Plot loss curves from ND-Uncertainty TensorBoard logs')
    parser.add_argument('--runs-dir', type=str, 
                       default='runs_unc_beta',
                       help='Directory containing run folders (default: runs_unc_beta)')
    parser.add_argument('--scan-ids', nargs='+', type=int,
                       default=None,
                       help='Scan IDs to process (default: process all found)')
    parser.add_argument('--base-dir', type=str,
                       default='/home/hussein/project/ND-Uncertainty',
                       help='Base directory (default: /home/hussein/project/ND-Uncertainty)')
    
    args = parser.parse_args()
    
    runs_dir = os.path.join(args.base_dir, args.runs_dir)
    
    if not os.path.exists(runs_dir):
        print(f"Error: Runs directory not found: {runs_dir}")
        return
    
    print("ND-Uncertainty Loss Plotter")
    print("="*60)
    
    # Find all scan IDs if not specified
    if args.scan_ids is None:
        # Find all replica_all8_* or replica_* directories
        scan_ids = []
        for item in os.listdir(runs_dir):
            if os.path.isdir(os.path.join(runs_dir, item)) and 'replica' in item:
                try:
                    # Handle both replica_all8_{id} and replica_{id} patterns
                    if 'replica_all8_' in item:
                        scan_id = int(item.split('_')[-1])
                    elif 'replica_' in item:
                        scan_id = int(item.split('_')[-1])
                    else:
                        continue
                    scan_ids.append(scan_id)
                except:
                    pass
        scan_ids = sorted(scan_ids)
        print(f"Found scan IDs: {scan_ids}")
    else:
        scan_ids = args.scan_ids
    
    # Process each scan and collect total loss and uncertainty loss data
    all_scans_total_loss = {}
    all_scans_uncertainty_loss = {}
    for scan_id in scan_ids:
        try:
            loss_data_dict = process_replica_run(runs_dir, scan_id, args.base_dir)
            if loss_data_dict is not None:
                if loss_data_dict.get('total') is not None:
                    all_scans_total_loss[scan_id] = loss_data_dict['total']
                if loss_data_dict.get('uncertainty_loss') is not None:
                    all_scans_uncertainty_loss[scan_id] = loss_data_dict['uncertainty_loss']
        except Exception as e:
            print(f"Error processing scan {scan_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create combined plot of all scans' total loss
    if all_scans_total_loss:
        print(f"\n{'='*60}")
        print("Creating combined total loss plot for all scans...")
        print(f"{'='*60}")
        combined_output_path = os.path.join(args.base_dir, 'all_losses', 'Replica', 'scan_total_loss.png')
        plot_combined_all_scans_total_loss(all_scans_total_loss, combined_output_path)
    
    # Create combined plot of all scans' uncertainty loss
    if all_scans_uncertainty_loss:
        print(f"\n{'='*60}")
        print("Creating combined uncertainty loss plot for all scans...")
        print(f"{'='*60}")
        combined_output_path = os.path.join(args.base_dir, 'all_losses', 'Replica', 'scan_uncertainty_loss.png')
        plot_combined_all_scans_uncertainty_loss(all_scans_uncertainty_loss, combined_output_path)
    
    # Create combined plot of total loss (blue) and uncertainty loss (red)
    if all_scans_total_loss or all_scans_uncertainty_loss:
        print(f"\n{'='*60}")
        print("Creating combined total + uncertainty loss plot...")
        print(f"{'='*60}")
        combined_output_path = os.path.join(args.base_dir, 'all_losses', 'Replica', 'full_total_unc_loss.png')
        plot_combined_total_and_uncertainty_loss(all_scans_total_loss, all_scans_uncertainty_loss, combined_output_path)
    
    print(f"\n{'='*60}")
    print("Done! Check the all_losses/Replica/ directory for plots.")
    print(f"  - Individual scans: all_losses/Replica/scan_1/ through scan_8/")
    print(f"  - Combined total loss: all_losses/Replica/scan_total_loss.png")
    print(f"  - Combined uncertainty loss: all_losses/Replica/scan_uncertainty_loss.png")
    print(f"  - Combined total + uncertainty: all_losses/Replica/full_total_unc_loss.png")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()

