# Scan 3 Training Failure & Uncertainty Visualization Issues

## Issues Identified

### 1. Scan 3 Training Failure
- **Problem**: Scan 3 stopped training at epoch 2016 (should continue to 2400)
- **Symptom**: Evaluation fails for scan 3; other scans (1, 2, 4) work fine
- **Possible Causes**:
  - Mesh extraction failed around epoch ~292, producing corrupted meshes
  - Training continued but meshes remained corrupted
  - Out-of-memory (OOM) error
  - NaN/Inf values causing training crash
  - Missing or corrupted data files for scan 3

### 2. Uncertainty Visualization - Lime Green Images
- **Problem**: Uncertainty images appear as solid lime green instead of showing variation
- **Root Cause**: All beta (sigma) values are being clamped to `sigma_max=0.5`
  - When MLP predicts log_sigma > log(0.5) ≈ -0.69, all values get clamped to 0.5
  - Uniform values → uniform color in visualization
- **Why Lime Green**: Turbo colormap at middle value (0.5) appears yellow/green

## Fixes Applied

### ✅ Indentation Fixes
1. **exp_runner.py**: Fixed indentation for:
   - `if dist.is_initialized():` blocks (lines 114-115, 210)
   - `quat/biased_normal` assignments (lines 309, 311, 313)

2. **evaluate_single_scene.py**: Fixed indentation for:
   - Path resolution (lines 26, 33)
   - Print statement in culling block (line 210)

### ✅ Uncertainty Visualization Fix
**File**: `nd_uncertainty/trainer.py` - `_save_uncertainty_heatmap()`

**Before**: Used fixed bounds `[0.2, 2.0]` which don't match actual sigma range `[1e-3, 0.5]`

**After**: 
- Clamp to actual sigma bounds `[sigma_min, sigma_max] = [1e-3, 0.5]`
- Normalize using actual min/max for better contrast
- Handles uniform values gracefully

## Diagnostic Steps

### Run Diagnostic Script
```bash
python diagnose_scan3.py
```

This will check:
1. Mesh file sizes and validity for scan 3
2. Uncertainty image statistics (uniformity check)
3. Scan 3 data file existence

### Check Training Logs
Look for errors around epoch 2016:
```bash
# Check log files in runs_new/replica_3/<timestamp>/logs/
grep -i "error\|nan\|oom\|crash" runs_new/replica_3/*/logs/*.txt
```

### Check Mesh Files
```bash
# Check if meshes are corrupted (should be >1KB, typically >100KB)
ls -lh runs_new/replica_3/*/plots/mesh_*.ply
```

## Next Steps

### For Uncertainty Visualization
The fix is in place, but if images are still uniform:
1. **Check MLP predictions**: If all log_sigma > -0.69, they'll all clamp to 0.5
2. **Possible causes**:
   - MLP not trained properly
   - Checkpoint from before sigma clamp fix
   - Training didn't converge for uncertainty branch
3. **Solution**: Re-train with fixed code (sigma clamping in loss_wrapper.py)

### For Scan 3 Training Failure
1. **If meshes are corrupted**:
   - Check training logs for errors
   - Verify data files exist (`data/Replica/scan3/`)
   - Re-train scan 3 from scratch with fixed code

2. **If data is missing**:
   - Verify `data/Replica/scan3/traj.txt` exists
   - Check `data/Replica/scan3/images/` has images
   - Verify cameras.npz exists

3. **If evaluation fails but training completed**:
   - Try evaluating specific epoch (e.g., earlier epoch if later ones corrupted)
   - Check if culling mesh script works: `python evals/replica_eval/cull_mesh.py ...`

## Code Verification

### Critical Fixes Confirmed
✅ `loss_wrapper.py` line 232: `sigma.clamp(min=self.sigma_min, max=self.sigma_max)`
✅ `uncertainty_mlp.py` line 104: `sigma.clamp(min=self.sigma_min, max=self.sigma_max)`
✅ `trainer.py` line 269-282: Uncertainty visualization uses actual sigma bounds

### Potential Issues
- If uncertainty images are still uniform, check if `render_beta_map()` returns all 0.5 values
- Verify MLP is training correctly (check loss curves in TensorBoard)
