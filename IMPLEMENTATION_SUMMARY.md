# ND-Uncertainty Implementation Summary

## Project Goal

Inject NeRF-on-the-Go's uncertainty mechanism into ND-SDF while keeping all existing ND-SDF behaviors intact. The system predicts per-ray uncertainty `β(r)` using a small MLP from DINO-based patch features, and uses a principled uncertainty loss from "NeRF in the Wild" to reweight RGB error.

## Core Architecture

The implementation follows a non-invasive approach: all new code lives in `nd_uncertainty/` package, and ND-SDF's core files remain untouched. Integration is achieved through:
- **Subclassing**: `UncertaintyTrainer` extends ND-SDF's `Trainer`
- **Wrapping**: `UncertaintyAwareLoss` wraps ND-SDF's `ImplicitReconLoss`
- **Pipeline**: `UncertaintyPipeline` orchestrates DINO → Patch → MLP → β(r)

## File Structure

```
nd_uncertainty/
├── __init__.py                    # Package initialization
├── dino_encoder.py                # DINOv2 feature extraction (with auto-padding)
├── patch_sampling.py               # Dilated patch sampling (memory-efficient indexing)
├── uncertainty_mlp.py              # Small MLP G(r) → β(r)
├── pipeline.py                     # Orchestrates DINO → Patch → MLP → β(r)
├── uncertainty_loss.py             # Core uncertainty loss + optional SSIM extension
├── variance_regularizer.py         # DINO patch similarity variance regularization
├── ssim_utils.py                   # SSIM components and bias function
├── loss_wrapper.py                 # Wraps ND-SDF loss, adds uncertainty terms
├── trainer.py                      # Subclasses ND-SDF trainer, injects β(r)
├── run_uncertainty.py              # Entry point with GPU routing
└── FINAL_IMPLEMENTATION_SUMMARY.md # Documentation
```

## Key Components

### 1. DINOv2 Encoder (`dino_encoder.py`)
- Loads pretrained DINOv2 ViT-B/14 via `torch.hub`
- Handles image padding to ensure dimensions are multiples of 14 (DINOv2 patch size)
- Returns feature maps: `(B, C, H_feat, W_feat)`
- Frozen by default (no gradients)

### 2. Dilated Patch Sampler (`patch_sampling.py`)
- Extracts localized patches from DINO feature maps corresponding to ray locations
- Uses dilated sampling (patch_size × patch_size with dilation spacing)
- **Memory-efficient implementation**: Direct advanced indexing per batch, avoiding large tensor expansions
- Handles coordinate mapping from image space to feature map space
- Returns patches: `(B, R, C * patch_size^2)`

### 3. Uncertainty MLP (`uncertainty_mlp.py`)
- Small MLP: `in_dim → hidden_dim → 1`
- Uses ReLU/softplus activations
- Outputs positive β(r) values (softplus + epsilon)
- **Lazy initialization**: Created on first forward pass when patch embedding dimension is known

### 4. Uncertainty Pipeline (`pipeline.py`)
- Orchestrates: RGB → DINO → Patch → MLP → β(r)
- Handles lazy MLP initialization
- Includes `render_beta_map()` method for full-image uncertainty visualization during validation

### 5. Uncertainty Loss (`uncertainty_loss.py`)
- Core loss (NeRF in the Wild): `L = ||C - Ĉ||² / (2β²) + λ log β`
- Optional SSIM-based uncertainty loss (from NeRF-on-the-Go)
- Numerical stability: clipping, epsilon, gradient stopping options

### 6. Patch Variance Regularizer (`variance_regularizer.py`)
- Encourages similar DINO patches to have similar predicted uncertainty
- Computes top-k patch similarity and regularizes variance

### 7. Loss Wrapper (`loss_wrapper.py`)
- Wraps ND-SDF's `ImplicitReconLoss`
- Computes native ND-SDF losses first
- Adds uncertainty loss and variance regularizer (if enabled)
- Implements training progress-based annealing for uncertainty weight
- **Critical**: Uses `__getattr__` delegation to forward attribute access to `base_loss` (e.g., `lambda_curvature`, `set_curvature_weight`)

### 8. Uncertainty Trainer (`trainer.py`)
- Subclasses ND-SDF's `Trainer` from `exp_runner.py`
- Overrides `prepare_sample()` to compute β(r) before ND-SDF forward pass
- Loads full-resolution RGB images for DINO processing
- Dynamically adds Uncertainty MLP parameters to optimizer after lazy initialization
- Overrides `plot()` to save uncertainty heatmaps during validation (every `plot_freq` epochs)
- **Uncertainty visualization**: Uses turbo colormap, fixed bounds (lo=0.2, hi=2), accumulation map weighting (matching NeRF-on-the-Go)

### 9. Entry Point (`run_uncertainty.py`)
- Replaces ND-SDF's `main.py` entry point
- Sets `CUDA_VISIBLE_DEVICES` internally (defaults to GPU 1, configurable)
- Includes GPU routing to avoid GPU 0 (Rahman) and GPU 3 (display)
- Sets `NCCL_P2P_DISABLE=1` to prevent NCCL from trying hidden GPUs
- Includes GPU info logging to verify which physical GPU is being used

## Configuration

### Config Files Created/Modified

1. **`confs/replica.yaml`**: Main config
   - Matches ND-SDF defaults: `batch_size=4`, `num_rays=1024`
   - `use_mask=false` (dataset JSON lacks `mask_path`)
   - Uncertainty loss settings in `loss:` section
   - Uncertainty pipeline settings in `uncertainty:` section

2. **`confs/replica_2gpu.yaml`**: 2-GPU training config
   - `batch_size=2` per GPU (effective batch = 4)
   - Same uncertainty settings as `replica.yaml`

3. **`confs/replica_all8.yaml`**: Config for running all 8 scans
   - Same as `replica.yaml` but with `exp_name: replica_all8`
   - Used by `run_all8_scans.sh` script

### Key Config Parameters

```yaml
loss:
  use_uncertainty: true
  use_ssim_uncertainty: false
  use_variance_regularizer: false
  weight_unc: 1.0
  unc_lambda_reg: 0.5
  unc_clip_min: 0.1
  unc_eps: 1e-3

uncertainty:
  patch_size: 7
  dilation: 2
  auto_tune: false
  max_chunk_rays: 16384
```

## Training Flow

1. **Ray Sampling**: ND-SDF samples rays (with deflection-angle guidance if enabled)
2. **Uncertainty Computation** (`trainer.py::compute_uncertainty`):
   - Loads full-resolution RGB images
   - Extracts DINOv2 features
   - Performs dilated patch sampling
   - Predicts β(r) via Uncertainty MLP
   - Attaches `beta` and `uncertainty_features` to `sample` dict
3. **ND-SDF Forward**: Normal ND-SDF forward pass (unchanged)
4. **Loss Computation** (`loss_wrapper.py`):
   - Computes native ND-SDF losses
   - Adds uncertainty-weighted color loss: `L_unc = ||C - Ĉ||² / (2β²) + λ log β`
   - Optionally adds variance regularizer
   - Applies training progress-based annealing to uncertainty weight
5. **Backpropagation**: Gradients flow through Uncertainty MLP (DINO encoder frozen)

## Visualization and Logging

### Uncertainty Heatmaps
- **Saved during validation** (every `plot_freq` epochs = every 240 epochs)
- **Location**: `runs_unc_beta/{exp_name}/{timestamp}/plots/uncertainty/epoch{epoch}_view{idx}.png`
- **Visualization style** (matching NeRF-on-the-Go):
  - Turbo colormap (not jet)
  - Fixed bounds: `lo=0.2, hi=2.0` (not min/max normalization)
  - Accumulation map weighting (background regions darkened)
- **Removed**: Training-time uncertainty saving (was creating too many images every 10 steps)

### TensorBoard Logging
- **Training-time**: `uncertainty/train_mean_beta`, `uncertainty/train_std_beta` (every `log_freq` steps)
- **Validation-time**: `uncertainty/val_mean_beta`, `uncertainty/val_median_beta`, `uncertainty/val_std_beta`, `uncertainty/val_min_beta`, `uncertainty/val_max_beta`, `uncertainty/val_p25_beta`, `uncertainty/val_p50_beta`, `uncertainty/val_p75_beta`, `uncertainty/val_p95_beta`
- **Histogram**: `uncertainty/beta_hist` (with try/except for older PyTorch versions)

## Memory Optimizations

### Patch Sampling Memory Efficiency
- **Problem**: Original `expand` + `gather` pattern created large intermediate tensors causing CUDA OOM on 4GB GPUs
- **Solution**: Direct advanced indexing loop per batch, avoiding tensor expansions
- **Code pattern**:
  ```python
  for b in range(B):
      feat_map_b = feature_maps[b]  # (C, H_feat, W_feat)
      sampled_b = feat_map_b[:, y_coords_b, x_coords_b]  # (C, R, num_patches)
  ```

### Config Adjustments for Small GPUs
- `patch_size: 7 → 3` (reduces patch feature dimension)
- `dilation: 2 → 1` (reduces memory footprint)
- `num_rays: 1024 → 512` (reduces batch size)
- `batch_size: 4 → 2` (reduces image batch)
- `chunk: 1024 → 512` (reduces rendering chunk size)
- **Note**: Auto-tuning was implemented but later disabled to match ND-SDF defaults

## GPU Routing

### Problem
- GPU 0 is occupied by Rahman
- GPU 3 is display GPU (not for compute)
- Need to use GPUs 1, 2, 4 (and potentially 5, 6, 7, 8 if available)

### Solution
- `run_uncertainty.py` sets `CUDA_VISIBLE_DEVICES` internally (defaults to GPU 1)
- When `CUDA_VISIBLE_DEVICES=1` is set, PyTorch remaps physical GPU 1 to logical "cuda:0"
- GPU info logging confirms which physical GPU is actually being used
- `NCCL_P2P_DISABLE=1` prevents NCCL from trying to communicate with hidden GPUs

## Multi-Scan Execution

### Script: `run_all8_scans.sh`
- Distributes 8 Replica scans across 3 GPUs:
  - GPU 1: scans 1, 4, 7
  - GPU 2: scans 2, 5, 8
  - GPU 4: scans 3, 6
- Each scan gets unique `master_port` (29525-29532)
- Launches all scans in parallel (background processes)
- Tracks PIDs for monitoring/killing

## Key Fixes and Resolutions

1. **`ModuleNotFoundError: tinycudann`**: Resolved via pip install with CUDA architecture flags
2. **`KeyError: 'mask_path'`**: Set `use_mask: false` in config
3. **`ValueError: Default process group not initialized`**: Made `DistributedSampler` and DDP conditional on `dist.is_initialized()`
4. **`TypeError: ImplicitReconLoss got unexpected keyword 'use_uncertainty'`**: Filtered uncertainty params before passing to `ImplicitReconLoss`
5. **`AssertionError: Input image height not multiple of 14`**: Added automatic padding in `dino_encoder.py`
6. **`NameError: patch_size not defined`**: Fixed to `self.patch_size`
7. **`IndexError: shape mismatch`**: Refactored to linear indexing
8. **`RuntimeError: expanded size mismatch`**: Fixed dilated patch offset generation to always create `patch_size^2` offsets
9. **`AssertionError: offset_range has 8 elements, expected 7`**: Fixed floor division for negative numbers
10. **`AttributeError: 'UncertaintyAwareLoss' has no attribute 'lambda_curvature'`**: Implemented robust `__getattr__` delegation
11. **`torch.OutOfMemoryError`**: Refactored patch sampling to memory-efficient advanced indexing, reduced config settings
12. **GPU routing confusion**: Added GPU info logging to show physical vs logical GPU mapping

## Loss Formula

The total loss is:
```
L_total = L_NDSDF_native + weight_unc * L_uncertainty + weight_reg * L_variance_reg
```

Where:
- `L_NDSDF_native`: All original ND-SDF losses (color, depth, normal, smoothness, curvature, etc.)
- `L_uncertainty = ||C - Ĉ||² / (2β²) + λ log β`: Uncertainty-weighted color loss
- `L_variance_reg`: Optional patch variance regularizer (disabled by default)

## Integration Points

1. **Before ND-SDF forward**: `UncertaintyTrainer.prepare_sample()` computes β(r)
2. **During loss computation**: `UncertaintyAwareLoss.forward()` adds uncertainty terms
3. **During validation**: `UncertaintyTrainer.plot()` saves uncertainty heatmaps
4. **During training**: Statistics logged to TensorBoard every `log_freq` steps

## Current Status

- ✅ Core uncertainty pipeline implemented and tested
- ✅ Training runs successfully on A100 (80GB)
- ✅ Uncertainty heatmaps saved during validation (matching NeRF-on-the-Go style)
- ✅ TensorBoard logging for uncertainty statistics
- ✅ GPU routing working correctly
- ✅ Config files match ND-SDF defaults (batch_size=4, num_rays=1024)
- ✅ Multi-scan execution script ready
- ⚠️ SSIM uncertainty loss implemented but disabled (can be enabled via config)
- ⚠️ Variance regularizer implemented but disabled (can be enabled via config)

## Usage

### Single Scan Training
```bash
torchrun --nproc_per_node=1 --master_port=29525 run_uncertainty.py \
    --conf confs/replica.yaml --scan_id 1 --data_dir '' --root_dir runs_unc_beta
```

### All 8 Scans (3 GPUs)
```bash
chmod +x run_all8_scans.sh
./run_all8_scans.sh
```

## Notes

- The uncertainty pipeline is **lazy**: Uncertainty MLP is created on first forward pass
- DINO encoder is **frozen**: No gradients flow through it
- Uncertainty MLP parameters are **dynamically added** to optimizer after initialization
- All uncertainty code is **isolated** in `nd_uncertainty/` package
- ND-SDF's core files (`exp_runner.py`, `models/system.py`, `models/loss.py`) remain **unchanged** (except for minor fixes like conditional DDP and parameter filtering)
