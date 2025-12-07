#!/bin/bash
# Run all 8 Replica scans distributed across 2 GPUs (2, 4)
# GPU 0 and 1 are occupied by another experiment
# Distribution: GPU 2 -> scans 1,2,3,4 (2 at a time) | GPU 4 -> scans 5,6,7,8 (2 at a time)

# GPU assignments: [gpu, scan_id] pairs
# GPU 2: scans 1, 2 (batch 1), then 3, 4 (batch 2)
# GPU 4: scans 5, 6 (batch 1), then 7, 8 (batch 2)

declare -A GPU_SCANS
GPU_SCANS[2]="1 2 3 4"
GPU_SCANS[4]="5 6 7 8"

PORT_BASE=29525
port_counter=0

# Function to launch a single scan
launch_scan() {
    local gpu=$1
    local scan_id=$2
    local master_port=$3
    
    echo "Launching scan_id: $scan_id on GPU $gpu (port $master_port)" >&2
    
    # Run in background
    # Set CUDA_DEVICE_ORDER=PCI_BUS_ID to match nvidia-smi device ordering
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu torchrun --nproc_per_node=1 --master_port=$master_port \
        run_uncertainty.py --conf confs/replica_all8.yaml --scan_id $scan_id --data_dir '' --root_dir runs_unc_beta &
    
    local pid=$!
    echo "Started scan $scan_id on GPU $gpu (PID: $pid, port: $master_port)" >&2
    echo $pid  # Return PID to stdout (for capture)
}

# Launch scans in batches of 2 per GPU
for batch in 1 2; do
    echo ""
    echo "=========================================="
    echo "Starting batch $batch (2 scans per GPU)"
    echo "=========================================="
    
    PIDS=()
    
    # Launch 2 scans per GPU in parallel
    for gpu in 2 4; do
        scans=(${GPU_SCANS[$gpu]})
        
        if [ $batch -eq 1 ]; then
            # Batch 1: first 2 scans
            scan1=${scans[0]}
            scan2=${scans[1]}
        else
            # Batch 2: last 2 scans
            scan1=${scans[2]}
            scan2=${scans[3]}
        fi
        
        echo "GPU $gpu will run scans: $scan1, $scan2"
        
        # Launch first scan
        master_port=$((PORT_BASE + port_counter))
        port_counter=$((port_counter + 1))
        pid1=$(launch_scan $gpu $scan1 $master_port)
        PIDS+=($pid1)
        sleep 2
        
        # Launch second scan
        master_port=$((PORT_BASE + port_counter))
        port_counter=$((port_counter + 1))
        pid2=$(launch_scan $gpu $scan2 $master_port)
        PIDS+=($pid2)
        sleep 2
    done
    
    echo ""
    echo "Batch $batch launched. PIDs: ${PIDS[@]}"
    echo "Waiting for batch $batch to complete..."
    
    # Wait for all processes in this batch to complete
    for pid in "${PIDS[@]}"; do
        # Check if process exists, wait for it
        if ps -p $pid > /dev/null 2>&1; then
            echo "Waiting for PID $pid..."
            wait $pid
            exit_code=$?
            if [ $exit_code -eq 0 ]; then
                echo "Scan with PID $pid completed successfully"
            else
                echo "Scan with PID $pid failed with exit code $exit_code"
            fi
        else
            # Process already exited
            wait $pid 2>/dev/null
            exit_code=$?
            echo "Scan with PID $pid already exited with code $exit_code"
        fi
    done
    
    echo "Batch $batch completed!"
done

echo ""
echo "=========================================="
echo "All 8 scans completed!"
echo "=========================================="
