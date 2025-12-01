#!/bin/bash
# Simple script to update existing nd_unc environment with new packages
# For workstation with CUDA 12

set -e

echo "Updating nd_unc environment..."

conda activate nd_unc

# Install/update packages needed for uncertainty components
pip install -r requirements.txt

# DINOv2 is loaded via torch.hub, no install needed
# Just make sure torch is available (should already be installed)

echo "Update complete!"
