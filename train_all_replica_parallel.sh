#!/bin/bash
# Train all 8 Replica scenes in parallel using 2 GPUs (1 scene per GPU)
# Faster overall but uses single-GPU config

cd /home/hussein/project/ND-SDF
conda activate nd_unc

echo "Starting parallel training for all 8 Replica scenes"
echo "GPU 1: scenes 1,3,5,7"
echo "GPU 2: scenes 2,4,6,8"
echo "Using config: replica_low.yaml (single-GPU training)"
echo "Estimated time: ~15.2 hours total"
echo ""

# Function to train on a specific GPU
train_on_gpu() {
    local gpu=$1
    local scan_id=$2
    CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 exp_runner.py \
        --conf confs/replica_low.yaml \
        --scan_id $scan_id \
        --data_dir ''
}

# Train scenes 1,3,5,7 on GPU 1 in background
for scan_id in 1 3 5 7; do
    echo "Starting scan_id $scan_id on GPU 1 at $(date)"
    train_on_gpu 1 $scan_id &
done

# Train scenes 2,4,6,8 on GPU 2 in background
for scan_id in 2 4 6 8; do
    echo "Starting scan_id $scan_id on GPU 2 at $(date)"
    train_on_gpu 2 $scan_id &
done

# Wait for all background jobs to complete
echo "Waiting for all training jobs to complete..."
wait

echo "=========================================="
echo "All 8 scenes completed!"
echo "End time: $(date)"
echo "=========================================="

