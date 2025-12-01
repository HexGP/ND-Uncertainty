"""
Entry point for ND-Uncertainty training.

This replaces exp_runner.py's main block to use UncertaintyTrainer.
"""

import os

# GPU Routing: Choose one A100 that is not GPU0 (Rahman) and not GPU3 (display).
# Change "1" to "2" or "4" if that's the free A100 you want to use.
# This sets CUDA_VISIBLE_DEVICES internally, so don't set it from command line.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change to 2 or 4 if needed

# Avoid NCCL trying to talk to hidden GPUs
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")

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
        # CUDA_VISIBLE_DEVICES is set at the top of this file
        # PyTorch will remap the selected GPU to logical "cuda:0"
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
