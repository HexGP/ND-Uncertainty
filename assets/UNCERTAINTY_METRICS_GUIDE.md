# Uncertainty Metrics Guide

## Problem: All Purple Uncertainty Visualization

**Current Issue:** Beta values are all ~1e-6, causing all-purple visualization.

**Root Cause:** 
- The uncertainty MLP learned to output very negative values during training
- Even with new initialization (bias=1.5), the model continues to output negative values
- This suggests the loss is pushing β towards minimum

## Required Uncertainty Metrics to Track

### 1. **Training-Time Metrics** (logged every `log_freq` steps)
- `uncertainty/beta_mean` - Average β value across training rays
- `uncertainty/beta_std` - Standard deviation of β
- `uncertainty/beta_p90` - 90th percentile
- `uncertainty/beta_p95` - 95th percentile
- `uncertainty/beta_hist` - Histogram of β values

### 2. **Validation-Time Metrics** (logged every `plot_freq` epochs)
- `uncertainty/val_mean_beta` - Average β on validation set
- `uncertainty/val_median_beta` - Median β
- `uncertainty/val_std_beta` - Standard deviation
- `uncertainty/val_min_beta` - Minimum β
- `uncertainty/val_max_beta` - Maximum β
- `uncertainty/val_p25_beta` - 25th percentile
- `uncertainty/val_p50_beta` - 50th percentile (median)
- `uncertainty/val_p75_beta` - 75th percentile
- `uncertainty/val_p95_beta` - 95th percentile

### 3. **Loss Metrics**
- `loss/uncertainty_loss` - Total uncertainty loss (L_unc)
- `loss/l_ssim` - Raw LSSIM value (Eq. 8)
- `loss/variance_regularizer` - Variance regularization term (L_reg)

## Expected Beta Value Ranges

Based on NeRF-on-the-Go and visualization bounds [0.2, 2.0]:

- **Low uncertainty (well-reconstructed):** β ≈ 0.2 - 0.5 (blue/purple in heatmap)
- **Medium uncertainty:** β ≈ 0.5 - 1.5 (green/yellow in heatmap)
- **High uncertainty (difficult regions):** β ≈ 1.5 - 2.0 (red/orange in heatmap)

**Current values:** All ~1e-6 ❌ (should be 0.2-2.0)

## What to Check

1. **Initial β values:** Should be ~0.5-1.5 after first forward pass
2. **β distribution:** Should have variation, not all at minimum
3. **β trends:** Should decrease in well-reconstructed regions, increase in difficult regions
4. **Uncertainty loss:** Should be negative (due to log term) but not too negative

## Solution

The bias=1.5 initialization fix needs a **fresh training run** to take effect. The current checkpoint was trained with bias=0, and the model learned to output very negative values.

**Next steps:**
1. Start fresh training with bias=1.5 initialization
2. Monitor beta_mean should start around 0.5-1.5, not 1e-6
3. Check that beta values have variation (std > 0.01)

