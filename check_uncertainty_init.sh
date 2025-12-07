#!/bin/bash
# Quick script to check if uncertainty initialization is working

echo "Checking uncertainty initialization..."
echo ""

# Wait for first log entry
sleep 5

# Check TensorBoard logs for beta values
LOG_DIR=$(find runs_test_unc -name "logs" -type d | head -1)

if [ -z "$LOG_DIR" ]; then
    echo "No logs found yet. Wait for training to start."
    exit 1
fi

echo "Found logs at: $LOG_DIR"
echo ""

# Use Python to read TensorBoard logs
source /home/hussein/miniconda3/etc/profile.d/conda.sh
conda activate nd_unc

python -c "
from tensorboard.backend.event_processing import event_accumulator
import os
import time

log_dir = '$LOG_DIR'
max_wait = 60  # Wait up to 60 seconds for logs
waited = 0

while waited < max_wait:
    try:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        if 'uncertainty/beta_mean' in ea.Tags()['scalars']:
            beta_events = ea.Scalars('uncertainty/beta_mean')
            if beta_events:
                first = beta_events[0]
                print(f'✅ First beta_mean at step {first.step}: {first.value:.6f}')
                
                if first.value > 0.1:
                    print(f'✅ GOOD: Beta is in reasonable range (not 1e-6)')
                    print(f'   Expected: 0.5-1.7, Got: {first.value:.6f}')
                else:
                    print(f'❌ BAD: Beta is too small (should be > 0.1)')
                    print(f'   Got: {first.value:.6f}')
                
                if len(beta_events) > 1:
                    last = beta_events[-1]
                    print(f'📊 Latest beta_mean at step {last.step}: {last.value:.6f}')
                break
    except:
        pass
    
    time.sleep(2)
    waited += 2
    if waited % 10 == 0:
        print(f'Waiting for logs... ({waited}s)')

if waited >= max_wait:
    print('Timeout: No beta values found in logs yet')
" 2>&1

