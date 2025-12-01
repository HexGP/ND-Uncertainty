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
        # When CUDA_VISIBLE_DEVICES=4 is set, PyTorch remaps GPU 4 to be "GPU 0"
        # So gpu=0 is correct - it will use the GPU specified in CUDA_VISIBLE_DEVICES
        gpu = 0
        device = torch.device("cuda", gpu)
        torch.cuda.set_device(device)
    
    # Verify which physical GPU we're actually using (works for both distributed and single GPU)
    physical_device_id = torch.cuda.current_device()
    device_props = torch.cuda.get_device_properties(physical_device_id)
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    print(f"[GPU Info] CUDA_VISIBLE_DEVICES={cuda_visible}")
    print(f"[GPU Info] PyTorch sees device {physical_device_id} as 'cuda:{physical_device_id}'")
    print(f"[GPU Info] Device name: {device_props.name}, Total memory: {device_props.total_memory / (1024**3):.2f} GB")
    
    # Create trainer with uncertainty pipeline
    trainer = UncertaintyTrainer(opt, gpu)
    
    # Start training
    trainer.train()
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
