#!/bin/bash
# Run all 8 Replica scans distributed across 3 GPUs (1, 2, 4)
# Distribution: GPU 1 -> scans 1,4,7 | GPU 2 -> scans 2,5,8 | GPU 4 -> scans 3,6

# GPU assignments: [gpu, scan_id] pairs
# GPU 1: scans 1, 4, 7
# GPU 2: scans 2, 5, 8
# GPU 4: scans 3, 6

declare -A GPU_SCANS
GPU_SCANS[1]="1 4 7"
GPU_SCANS[2]="2 5 8"
GPU_SCANS[4]="3 6"

# Array to store background process PIDs
PIDS=()
PORT_BASE=29525
port_counter=0

# Launch scans for each GPU
for gpu in 1 2 4; do
    scans=${GPU_SCANS[$gpu]}
    echo "GPU $gpu will run scans: $scans"
    
    for scan_id in $scans; do
        master_port=$((PORT_BASE + port_counter))
        port_counter=$((port_counter + 1))
        
        echo "Launching scan_id: $scan_id on GPU $gpu (port $master_port)"
        
        # Run in background
        CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$master_port \
            run_uncertainty.py --conf confs/replica_all8.yaml --scan_id $scan_id --data_dir '' --root_dir runs_unc_beta &
        
        PIDS+=($!)
        echo "Started scan $scan_id on GPU $gpu (PID: ${PIDS[-1]}, port: $master_port)"
        
        # Small delay to avoid port conflicts
        sleep 2
    done
done

echo ""
echo "All 8 scans launched across 3 GPUs. PIDs: ${PIDS[@]}"
echo "GPU 1: scans 1, 4, 7"
echo "GPU 2: scans 2, 5, 8"
echo "GPU 4: scans 3, 6"
echo ""
echo "Monitor with: watch -n 1 'ps aux | grep run_uncertainty'"
echo "Kill all with: kill ${PIDS[@]}"

# Wait for all processes to complete
wait

echo "All scans completed!"
