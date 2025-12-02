#!/bin/bash
# Run all 8 Replica scans simultaneously - one scan per GPU
# GPUs used: 1, 2, 4, 5, 6, 7, 8 (avoiding GPU 0 for Rahman, GPU 3 for display)

# Array of GPU IDs to use (8 GPUs for 8 scans)
GPUS=(1 2 4 5 6 7 8 1)  # Using GPU 1 twice if you only have 7 free GPUs, or adjust as needed

# Array of scan IDs (1-8)
SCAN_IDS=(1 2 3 4 5 6 7 8)

# Array to store background process PIDs
PIDS=()

# Launch all 8 scans in parallel
for i in {0..7}; do
    scan_id=${SCAN_IDS[$i]}
    gpu=${GPUS[$i]}
    
    echo "Launching scan_id: $scan_id on GPU $gpu"
    
    # Run in background
    CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$((29525 + i)) \
        run_uncertainty.py --conf confs/replica_all8.yaml --scan_id $scan_id --data_dir '' --root_dir runs_unc_beta &
    
    PIDS+=($!)
    echo "Started scan $scan_id on GPU $gpu (PID: ${PIDS[$i]})"
    
    # Small delay to avoid port conflicts
    sleep 2
done

echo ""
echo "All 8 scans launched. PIDs: ${PIDS[@]}"
echo "Monitor with: watch -n 1 'ps aux | grep run_uncertainty'"
echo "Kill all with: kill ${PIDS[@]}"

# Wait for all processes to complete
wait

echo "All scans completed!"
