# Architecture Changes: Heteroscedastic Uncertainty for Color Loss Only

## Summary of Changes

Based on the ND-SDF principles document, we need to implement **heteroscedastic uncertainty ONLY for color/photometric loss**, while keeping SDF, eikonal, and normal losses **deterministic** (no uncertainty).

## Current Implementation Issues

1. **RGB loss is standard L1** - not uncertainty-weighted
2. **Uncertainty loss is separate** - added to total, not replacing RGB loss
3. **SSIM-based approach** - not matching the heteroscedastic formulation
4. **All losses could be uncertainty-weighted** - violates the principle

## Required Changes

### 1. New Heteroscedastic Color Loss Function

**Formula from image:**
```
L_color(r) = (1 / (2 * σ_c(r)^2)) * ||C(r) - Ĉ(r)||^2 + (1/2) * log(σ_c(r)^2)
```

Where:
- `C(r)` = ground-truth color for ray `r`
- `Ĉ(r)` = rendered color for ray `r`
- `σ_c(r)` = predicted uncertainty for ray `r` (from uncertainty MLP)

**Implementation:**
- Create `heteroscedastic_color_loss()` function
- Input: `rgb_pred`, `rgb_gt`, `sigma` (uncertainty), `mask`
- Output: per-ray loss, then averaged

### 2. Regularizer on σ

**Formula from image:**
```
R(σ) = (1/N) * Σ_r (log σ_c(r) - log σ_0)^2
```

Where:
- `N` = number of rays
- `σ_0` = baseline uncertainty (e.g., 0.1 or configurable)
- `β` = `unc_lambda_reg` (weight for regularizer)

**Implementation:**
- Create `uncertainty_regularizer()` function
- Add to total loss as `β * R(σ)`

### 3. Modify Loss Wrapper

**Changes to `loss_wrapper.py`:**

1. **Replace RGB L1 loss** with heteroscedastic color loss when uncertainty is enabled
2. **Keep SDF, eikonal, normal losses deterministic** (no changes needed - they're already deterministic)
3. **Map parameters:**
   - `weight_unc` → `λ_color` (weight for heteroscedastic color loss)
   - `unc_lambda_reg` → `β` (weight for regularizer)

**New total loss formula:**
```
L = λ_sdf * L_SDF + λ_eik * L_eik + λ_normal * L_normal + λ_color * Σ_r L_color(r) + β * R(σ)
```

### 4. Remove SSIM-based Uncertainty Loss

- The current SSIM-based approach should be replaced with heteroscedastic formulation
- Keep SSIM computation optional for logging/monitoring only

## Implementation Steps

1. ✅ Create `heteroscedastic_color_loss()` in `nd_uncertainty/uncertainty_loss.py`
2. ✅ Create `uncertainty_regularizer()` in `nd_uncertainty/uncertainty_loss.py`
3. ✅ Modify `loss_wrapper.py` to:
   - Replace RGB L1 with heteroscedastic color loss
   - Add regularizer to total loss
   - Keep all other losses unchanged
4. ✅ Update configuration comments to reflect new formulation
5. ✅ Test with existing configs

## Parameter Mapping

| Config Parameter | Formula Symbol | Purpose |
|-----------------|----------------|---------|
| `weight_unc` | `λ_color` | Weight for heteroscedastic color loss |
| `unc_lambda_reg` | `β` | Weight for uncertainty regularizer |
| `unc_clip_min` | `σ_0` (baseline) | Minimum uncertainty / baseline for regularizer |

## Expected Benefits

1. **Stability**: Heteroscedastic formulation is more stable than SSIM-based approach
2. **Correctness**: Matches ND-SDF principles (uncertainty only for color)
3. **No spikes**: Should eliminate the instability spikes seen with `weight_unc=0.5`
4. **Better convergence**: Proper uncertainty weighting for color errors

