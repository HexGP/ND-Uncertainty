"""
Entry point for ND-Uncertainty training.

This replaces exp_runner.py's main block to use UncertaintyTrainer.
"""

import os
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
        torch.cuda.set_device(gpu)
    
    # Create trainer with uncertainty pipeline
    trainer = UncertaintyTrainer(opt, gpu)
    
    # Start training
    trainer.train()
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
