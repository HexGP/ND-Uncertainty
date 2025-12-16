# Heteroscedastic Uncertainty Implementation

## Summary

Implemented heteroscedastic uncertainty for color loss only, per ND-SDF principles. This replaces the previous SSIM-based approach with a more stable formulation.

## Changes Made

### 1. New Functions in `uncertainty_loss.py`

#### `heteroscedastic_color_loss(rgb_pred, rgb_gt, sigma, mask=None)`
- **Formula**: `L_color(r) = (1 / (2 * σ_c(r)^2)) * ||C(r) - Ĉ(r)||^2 + (1/2) * log(σ_c(r)^2)`
- Replaces standard RGB L1 loss when uncertainty is enabled
- Down-weights color errors when uncertainty is high
- Prevents uncertainty from collapsing to zero via log term

#### `uncertainty_regularizer(sigma, sigma_0=0.1, mask=None)`
- **Formula**: `R(σ) = (1/N) * Σ_r (log σ_c(r) - log σ_0)^2`
- Encourages uncertainty to stay close to baseline σ_0
- Prevents trivial inflation of uncertainty values

### 2. Modified `loss_wrapper.py`

#### Key Changes:
1. **Replaces RGB L1 with heteroscedastic color loss**
   - When uncertainty is enabled, standard `rgb_l1` loss is replaced
   - Uses `weight_unc` as `λ_color` (weight for heteroscedastic color loss)

2. **Adds uncertainty regularizer**
   - Uses `unc_lambda_reg` as `β` (weight for regularizer)
   - Added to total loss: `β * R(σ)`

3. **Keeps SDF, eikonal, normal losses deterministic**
   - These losses remain unchanged (no uncertainty weighting)
   - Follows ND-SDF principle: uncertainty ONLY for color

#### New Total Loss Formula:
```
L = λ_sdf * L_SDF + λ_eik * L_eik + λ_normal * L_normal + λ_color * Σ_r L_color(r) + β * R(σ)
```

Where:
- `λ_color` = `weight_unc` (from config)
- `β` = `unc_lambda_reg` (from config)
- `L_color(r)` = heteroscedastic color loss per ray
- `R(σ)` = uncertainty regularizer

## Parameter Mapping

| Config Parameter | Formula Symbol | Purpose |
|-----------------|----------------|---------|
| `weight_unc` | `λ_color` | Weight for heteroscedastic color loss |
| `unc_lambda_reg` | `β` | Weight for uncertainty regularizer |
| `unc_clip_min` | `σ_0` | Baseline uncertainty for regularizer |

## Backward Compatibility

- Legacy SSIM-based uncertainty loss is still available if `use_ssim_uncertainty: true` is set
- Default is heteroscedastic formulation (`use_ssim_uncertainty: false`)
- All existing configs will automatically use the new heteroscedastic formulation

## Expected Benefits

1. **Stability**: Heteroscedastic formulation is more stable than SSIM-based approach
2. **Correctness**: Matches ND-SDF principles (uncertainty only for color)
3. **No spikes**: Should eliminate instability spikes seen with `weight_unc=0.5`
4. **Better convergence**: Proper uncertainty weighting for color errors only

## Testing

The implementation should work with existing configs:
- `replica_w01_lam01.yaml` (weight=0.1, lambda=0.1)
- `replica_w01_lam05.yaml` (weight=0.1, lambda=0.5)
- `replica_w05_lam01.yaml` (weight=0.5, lambda=0.1)
- `replica_w05_lam05.yaml` (weight=0.5, lambda=0.5)

All configs will now use heteroscedastic color loss instead of SSIM-based uncertainty loss.

