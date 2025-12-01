"""
Entry point for ND-Uncertainty training.

This replaces exp_runner.py's main block to use UncertaintyTrainer.
"""

# NCCL_P2P_DISABLE: Force NCCL to stop trying GPU 0
import os
os.environ.setdefault("NCCL_P2P_DISABLE", "1")

# Optionally pin to a single visible GPU if not already constrained by the job manager
# For example, to default to GPU 4 (can be overridden by CUDA_VISIBLE_DEVICES env var):
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

import sys
import torch
import torch.distributed as dist
from exp_runner import get_args, init_processes
from nd_uncertainty.trainer import UncertaintyTrainer


def main():
    opt = get_args()
    
    # Initialize distributed training if needed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        gpu = init_processes()
    else:
        # Single GPU training
        gpu = 0
        device = torch.device("cuda", gpu)
        torch.cuda.set_device(device)
    
    # Create trainer with uncertainty pipeline
    trainer = UncertaintyTrainer(opt, gpu)
    
    # Start training
    trainer.train()
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
