#!/bin/bash
# Test script to run training from beginning (2000 steps quick test)
# Tests scan 1 on GPU 2 with replica_all8_test.yaml config

SCAN_ID=1
GPU=2
PORT=29525

echo "=========================================="
echo "Testing training from beginning (2000 steps)"
echo "=========================================="
echo "Scan ID: $SCAN_ID"
echo "GPU: $GPU"
echo "Port: $PORT"
echo "Config: confs/replica_all8_test.yaml"
echo ""

# Run from beginning (no --is_continue flag)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc_per_node=1 --master_port=$PORT \
    run_uncertainty.py --conf confs/replica_all8_test.yaml --scan_id $SCAN_ID --data_dir '' --root_dir runs_test_unc

echo ""
echo "Test completed!"

