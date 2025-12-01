"""
Ablation Study Runner

Runs three training modes to compare:
A: Baseline ND-SDF (no uncertainty)
B: ND-SDF + β(r) uncertainty only
C: ND-SDF + β(r) + SSIM + variance regularizer

Limited to 1-3 epochs for quick testing on 3090.
"""

import os
import sys
import torch
import torch.distributed as dist
import argparse
from omegaconf import OmegaConf
from exp_runner import get_args, init_processes
from nd_uncertainty.trainer import UncertaintyTrainer


def create_ablation_config(base_conf_path, mode):
    """
    Create config for ablation mode.
    
    Args:
        base_conf_path: Path to base config file
        mode: 'baseline', 'beta', or 'full'
    
    Returns:
        Modified config object
    """
    conf = OmegaConf.load(base_conf_path)
    
    # Limit training for quick ablation (1-3 epochs)
    # Estimate: ~1000 steps per epoch for Replica
    conf.train.max_step = min(conf.train.max_step, 3000)  # ~3 epochs max
    conf.train.plot_freq = 1000  # Plot more frequently for ablation
    conf.train.save_freq = 1000
    
    if mode == 'baseline':
        # Disable uncertainty completely
        conf.loss.use_uncertainty = False
        conf.loss.use_ssim_uncertainty = False
        conf.loss.use_variance_regularizer = False
        print("=" * 60)
        print("MODE A: Baseline ND-SDF (no uncertainty)")
        print("=" * 60)
        
    elif mode == 'beta':
        # Only basic uncertainty loss
        conf.loss.use_uncertainty = True
        conf.loss.use_ssim_uncertainty = False
        conf.loss.use_variance_regularizer = False
        print("=" * 60)
        print("MODE B: ND-SDF + β(r) uncertainty only")
        print("=" * 60)
        
    elif mode == 'full':
        # Full uncertainty system
        conf.loss.use_uncertainty = True
        conf.loss.use_ssim_uncertainty = True
        conf.loss.use_variance_regularizer = True
        conf.loss.use_uncertainty_annealing = True
        print("=" * 60)
        print("MODE C: ND-SDF + β(r) + SSIM + variance regularizer")
        print("=" * 60)
        
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'baseline', 'beta', or 'full'")
    
    return conf


def main():
    parser = argparse.ArgumentParser(description='Run ablation study for ND-Uncertainty')
    parser.add_argument('--conf', type=str, default='confs/replica.yaml', help='Base config file')
    parser.add_argument('--mode', type=str, required=True, choices=['baseline', 'beta', 'full'],
                       help='Ablation mode: baseline, beta, or full')
    parser.add_argument('--data_dir', type=str, default='', help='Data directory override')
    parser.add_argument('--scan_id', type=str, default='-1', help='Scan ID override')
    parser.add_argument('--root_dir', type=str, default='runs_ablation', help='Output root directory')
    parser.add_argument('--epoches', type=int, default=3, help='Number of epochs (limited for ablation)')
    
    # Parse args
    opt = parser.parse_args()
    
    # Create ablation config
    conf = create_ablation_config(opt.conf, opt.mode)
    
    # Override data settings if provided
    if opt.data_dir:
        conf.dataset.data_dir = opt.data_dir
    if opt.scan_id != '-1':
        conf.dataset.scan_id = opt.scan_id
    
    # Modify exp name to include mode
    original_exp_name = conf.train.exp_name
    conf.train.exp_name = f"{original_exp_name}_{opt.mode}"
    
    # Create modified opt object for trainer
    # Need to save config to file temporarily so Trainer can load it
    import tempfile
    import shutil
    temp_conf_path = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    OmegaConf.save(conf, temp_conf_path.name)
    temp_conf_path.close()
    
    class AblationOpt:
        def __init__(self, conf_path, opt):
            self.conf = conf_path  # Path to config file
            self.data_dir = opt.data_dir
            self.scan_id = opt.scan_id
            self.root_dir = opt.root_dir
            self.epoches = opt.epoches
            self.is_continue = False
            self.checkpoint = 'latest'
    
    ablation_opt = AblationOpt(temp_conf_path.name, opt)
    
    # Initialize distributed training if needed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        gpu = init_processes()
    else:
        # Single GPU training
        gpu = 0
        torch.cuda.set_device(gpu)
    
    # Always use UncertaintyTrainer for consistency
    # It will skip uncertainty computation if use_uncertainty=False
    trainer = UncertaintyTrainer(ablation_opt, gpu)
    
    print(f"\nStarting ablation training: {opt.mode}")
    print(f"Config: {opt.conf}")
    print(f"Data: {conf.dataset.data_dir}")
    print(f"Max steps: {conf.train.max_step}")
    print(f"Epochs: {opt.epoches}")
    print()
    
    # Start training
    trainer.train()
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()
    
    print(f"\nAblation training complete: {opt.mode}")
    
    # Cleanup temp config file
    try:
        os.unlink(temp_conf_path.name)
    except:
        pass


if __name__ == '__main__':
    main()
