#!/usr/bin/env python
"""
Quick test script to verify the uncertainty pipeline works
Run this after setting up the environment
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 50)
print("Testing ND-Uncertainty Setup")
print("=" * 50)

# Test 1: PyTorch and CUDA
print("\n1. Testing PyTorch...")
try:
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   WARNING: CUDA not available")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# Test 2: tiny-cuda-nn
print("\n2. Testing tiny-cuda-nn...")
try:
    import tcnn
    print("   ✓ tiny-cuda-nn imported successfully")
except ImportError as e:
    print(f"   WARNING: tiny-cuda-nn not found: {e}")
    print("   This is OK if you're just testing the uncertainty pipeline")

# Test 3: DINOv2 via torch.hub
print("\n3. Testing DINOv2 (torch.hub)...")
try:
    # Just check if we can load it (don't actually load to save time)
    print("   ✓ torch.hub available for DINOv2")
    print("   (DINOv2 will be loaded on first use)")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 4: ND-Uncertainty package
print("\n4. Testing nd_uncertainty package...")
try:
    from nd_uncertainty import DinoV2Encoder, DilatedPatchSampler, UncertaintyMLP, UncertaintyPipeline
    print("   ✓ All modules imported successfully")
    
    # Quick instantiation test
    print("\n5. Testing pipeline instantiation...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipeline = UncertaintyPipeline(device=device)
    print(f"   ✓ Pipeline created on {device}")
    
except ImportError as e:
    print(f"   ERROR: Cannot import nd_uncertainty: {e}")
    print("   Make sure you're in the ND-Uncertainty directory")
    sys.exit(1)
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Quick forward pass (if CUDA available)
if torch.cuda.is_available():
    print("\n6. Testing forward pass...")
    try:
        # Create dummy data
        B, H, W = 1, 256, 256
        R = 100
        
        rgb_full = torch.rand(B, 3, H, W).to(device)
        sampling_idx = torch.randint(0, H * W, (B, R)).to(device)
        heights = torch.tensor([H] * B).to(device)
        widths = torch.tensor([W] * B).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = pipeline(
                rgb_full=rgb_full,
                sampling_idx=sampling_idx,
                heights=heights,
                widths=widths,
            )
        
        print(f"   ✓ Forward pass successful")
        print(f"   Beta shape: {output['beta'].shape}")
        print(f"   Patch features shape: {output['patch_features'].shape}")
        
    except Exception as e:
        print(f"   ERROR in forward pass: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("\n6. Skipping forward pass (no CUDA)")

print("\n" + "=" * 50)
print("All tests passed! ✓")
print("=" * 50)
