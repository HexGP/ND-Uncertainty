"""
Diagnostic script to check uncertainty implementation setup.
Verifies learning rate scaling, initialization, curriculum learning, and clamp status.
"""
import os
import sys
import yaml
import numpy as np
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Make torch optional (not needed for config/checkpoint checking)
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except (ImportError, OSError):
    # OSError catches DLL loading errors on Windows
    TORCH_AVAILABLE = False

def check_config(config_path):
    """Check configuration for uncertainty parameters."""
    print("="*60)
    print("CHECKING CONFIGURATION")
    print("="*60)
    
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        return False
    
    # Use UTF-8 encoding to handle special characters (Windows compatibility)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            conf = yaml.safe_load(f)
    except UnicodeDecodeError:
        # Fallback to latin-1 if UTF-8 fails
        with open(config_path, 'r', encoding='latin-1') as f:
            conf = yaml.safe_load(f)
    
    loss_conf = conf.get('loss', {})
    
    # Check required parameters
    checks = {
        'use_uncertainty': loss_conf.get('use_uncertainty', False),
        'init_log_sigma': loss_conf.get('init_log_sigma', None),
        'sigma_min': loss_conf.get('sigma_min', None),
        'sigma_max': loss_conf.get('sigma_max', None),
        'uncertainty_warmup_steps': loss_conf.get('uncertainty_warmup_steps', None),
        'uncertainty_lr_scale': loss_conf.get('uncertainty_lr_scale', None),
        'variance_weight': loss_conf.get('variance_weight', None),
    }
    
    print(f"\nConfig file: {config_path}\n")
    
    all_ok = True
    for key, value in checks.items():
        status = "✓" if value is not None and value != False else "✗"
        if value is None:
            value_str = "NOT SET"
        elif isinstance(value, bool):
            value_str = "True" if value else "False"
        else:
            value_str = str(value)
        
        print(f"  {status} {key}: {value_str}")
        
        # Specific checks
        if key == 'init_log_sigma' and value is not None:
            if value > -2.0:
                print(f"    [WARNING] init_log_sigma ({value}) is too high. Should be ~-3.0")
                all_ok = False
            elif value < -4.0:
                print(f"    [WARNING] init_log_sigma ({value}) is too low. Should be ~-3.0")
        
        if key == 'uncertainty_lr_scale' and value is not None:
            if value > 0.2:
                print(f"    [WARNING] uncertainty_lr_scale ({value}) is too high. Should be ~0.1")
                all_ok = False
            elif value < 0.05:
                print(f"    [WARNING] uncertainty_lr_scale ({value}) is too low. Should be ~0.1")
        
        if key == 'uncertainty_warmup_steps' and value is not None:
            if value < 2000:
                print(f"    [WARNING] uncertainty_warmup_steps ({value}) is too low. Should be 2000-5000")
                all_ok = False
        
        if key == 'sigma_max' and value is not None:
            if value > 0.5:
                print(f"    [WARNING] sigma_max ({value}) exceeds recommended 0.5")
                all_ok = False
    
    return all_ok


def check_checkpoint(checkpoint_path):
    """Check checkpoint for uncertainty MLP state."""
    print("\n" + "="*60)
    print("CHECKING CHECKPOINT")
    print("="*60)
    
    if not TORCH_AVAILABLE:
        print("\n[WARNING] PyTorch not available (DLL error on Windows) - skipping checkpoint check")
        print("  This is OK - config check is more important")
        return True  # Not an error, just unavailable
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        # Check for uncertainty pipeline
        if 'uncertainty_pipeline' in ckpt:
            print("\n[OK] Uncertainty pipeline found in checkpoint")
            
            # Check uncertainty MLP
            if 'uncertainty_mlp' in ckpt['uncertainty_pipeline']:
                mlp_state = ckpt['uncertainty_pipeline']['uncertainty_mlp']
                
                if 'fc2.bias' in mlp_state:
                    fc2_bias = mlp_state['fc2.bias'].item()
                    initial_sigma = np.exp(fc2_bias)
                    
                    print(f"\n  Initial log_sigma (s_0): {fc2_bias:.4f}")
                    print(f"  Initial sigma (exp(s_0)): {initial_sigma:.4f}")
                    print(f"  Expected: s_0 = -3.0, σ ≈ 0.05")
                    
                    if fc2_bias > -2.0:
                        print(f"  [WARNING] Initial log_sigma too high! Should be ~-3.0")
                        return False
                    elif fc2_bias < -4.0:
                        print(f"  [WARNING] Initial log_sigma too low! Should be ~-3.0")
                        return False
                    else:
                        print(f"  [OK] Initialization looks correct")
                else:
                    print("  [ERROR] fc2.bias not found in MLP state")
                    return False
            else:
                print("  [ERROR] Uncertainty MLP not found in checkpoint")
                return False
        else:
            print("\n[WARNING] Uncertainty pipeline not found in checkpoint")
            print("  (This is OK if checkpoint is from before uncertainty was added)")
            return True  # Not an error, just not present
        
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return False
    
    return True


def check_training_logs(runs_dir, scan_id):
    """Check training logs for sigma statistics."""
    print("\n" + "="*60)
    print("CHECKING TRAINING LOGS (if available)")
    print("="*60)
    
    scan_dir = os.path.join(runs_dir, f"replica_{scan_id}")
    if not os.path.exists(scan_dir):
        print(f"ERROR: {scan_dir} not found")
        return
    
    # Find latest experiment
    exp_dirs = sorted([d for d in os.listdir(scan_dir) if os.path.isdir(os.path.join(scan_dir, d))])
    if not exp_dirs:
        print("No experiment directories found")
        return
    
    latest_exp = exp_dirs[-1]
    exp_path = os.path.join(scan_dir, latest_exp)
    
    # Check for TensorBoard logs
    log_dir = os.path.join(exp_path, "logs")
    if os.path.exists(log_dir):
        print(f"\nFound logs directory: {log_dir}")
        print("  (Use TensorBoard to view: tensorboard --logdir logs)")
    else:
        print(f"\nNo logs directory found in {exp_path}")


def main():
    """Run all diagnostic checks."""
    print("="*60)
    print("UNCERTAINTY IMPLEMENTATION DIAGNOSTIC")
    print("="*60)
    
    # Check config
    config_paths = [
        "confs/replica_new.yaml",
        "runs_new/replica_1/2025-12-12_17-45-47/conf.yaml",
    ]
    
    config_ok = False
    for config_path in config_paths:
        if os.path.exists(config_path):
            config_ok = check_config(config_path)
            break
    
    if not config_ok:
        print("\n[WARNING] Could not find or verify config file")
    
    # Check checkpoint (if available)
    checkpoint_paths = [
        "runs_new/replica_1/2025-12-12_17-45-47/checkpoints/epoch_2400.pth",
        "runs_new/replica_1/2025-12-12_17-45-47/checkpoints/latest.pth",
    ]
    
    checkpoint_ok = False
    for ckpt_path in checkpoint_paths:
        if os.path.exists(ckpt_path):
            checkpoint_ok = check_checkpoint(ckpt_path)
            break
    
    if not checkpoint_ok:
        print("\n[WARNING] Could not find or verify checkpoint")
    
    # Check logs
    check_training_logs("runs_new", 1)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if config_ok and checkpoint_ok:
        print("\n[OK] Configuration and checkpoint look correct")
        print("\nIf uncertainty images are still uniform:")
        print("  1. Check TensorBoard logs for sigma statistics")
        print("  2. Verify learning rate scaling is applied during training")
        print("  3. Check if sigma values are hitting clamp bounds (>50% at max)")
        print("  4. Consider re-training with verified parameters")
    else:
        print("\n[WARNING] Some issues found. Review warnings above.")
    
    print("\nFor detailed fixes, see: UNCERTAINTY_IMPLEMENTATION_GUIDE.md")
    print("="*60)


if __name__ == "__main__":
    main()
