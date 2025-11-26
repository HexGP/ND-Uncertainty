#!/bin/bash
# Train all 8 Replica scenes using 2 GPUs (GPU 1 & 2) with replica_2gpu.yaml config
# This matches the paper's effective batch size

cd /home/hussein/project/ND-SDF
conda activate nd_unc

# Use GPU 1 and 2
export CUDA_VISIBLE_DEVICES=1,2

echo "Starting training for all 8 Replica scenes on GPUs 1 & 2"
echo "Using config: replica_2gpu.yaml (2-GPU training, matches paper)"
echo "Estimated time: ~16.4 hours total"
echo ""

for scan_id in {1..8}; do
    echo "=========================================="
    echo "Training scan_id: $scan_id / 8"
    echo "Start time: $(date)"
    echo "=========================================="
    
    torchrun --nproc_per_node=2 exp_runner.py \
        --conf confs/replica_2gpu.yaml \
        --scan_id $scan_id \
        --data_dir ''
    
    echo "Completed scan_id: $scan_id at $(date)"
    echo ""
done

echo "=========================================="
echo "All 8 scenes completed!"
echo "End time: $(date)"
echo "=========================================="

