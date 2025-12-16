# Implementation Checklist - Heteroscedastic Uncertainty

## ✅ Completed Implementation

### 1. Uncertainty MLP (`uncertainty_mlp.py`)
- ✅ Predicts log-variance `s = log σ` instead of `σ` directly
- ✅ Computes `σ = exp(s)` for numerical stability
- ✅ Clamps `σ` to `[1e-3, 0.5]` to prevent extreme down-weighting
- ✅ Initializes `log σ` to `s_0 = -3` (σ ≈ 0.05) to prevent underfitting

### 2. Heteroscedastic Color Loss (`uncertainty_loss.py`)
- ✅ Implements `L_color(r) = (1/(2σ²)) * ||C - Ĉ||² + (1/2) * log(σ²)`
- ✅ Replaces standard RGB L1 loss when uncertainty is enabled
- ✅ Handles masking correctly

### 3. Uncertainty Regularizer (`uncertainty_loss.py`)
- ✅ Implements `R(σ) = (1/N) * Σ_r (log σ - log σ_0)²`
- ✅ Uses `σ_0 = exp(s_0) = exp(-3) ≈ 0.05` as baseline

### 4. Loss Wrapper (`loss_wrapper.py`)
- ✅ Replaces RGB L1 with heteroscedastic color loss
- ✅ Adds uncertainty regularizer to total loss
- ✅ Keeps SDF, eikonal, normal losses deterministic (no uncertainty)
- ✅ Implements curriculum learning: warmup stage (N_warm = 5000 steps default)
- ✅ Default hyperparameters:
  - `λ_color = 1.0` (weight_unc)
  - `β = 0.01` (unc_lambda_reg)
  - `s_0 = -3` (init_log_sigma)
  - `σ ∈ [1e-3, 0.5]` (sigma_min, sigma_max)

### 5. Trainer (`trainer.py`)
- ✅ Sets uncertainty MLP learning rate to 0.1x color LR (uncertainty_lr_scale = 0.1)
- ✅ Passes initialization parameters (s_0, σ_min, σ_max) to pipeline

### 6. Pipeline (`pipeline.py`)
- ✅ Passes initialization parameters to UncertaintyMLP
- ✅ Configurable `init_log_sigma`, `sigma_min`, `sigma_max`

### 7. Training Loop (`exp_runner.py`)
- ✅ Passes `cur_step` to loss forward for curriculum learning

## Default Hyperparameters (ND-SDF Recommended)

| Parameter | Symbol | Default Value | Config Key |
|-----------|--------|---------------|------------|
| Color loss weight | λ_color | 1.0 | `weight_unc` |
| Regularizer weight | β | 0.01 | `unc_lambda_reg` |
| Initial log σ | s_0 | -3.0 | `init_log_sigma` |
| Initial σ | σ_0 | ≈ 0.05 | `exp(init_log_sigma)` |
| σ minimum | σ_min | 1e-3 | `sigma_min` |
| σ maximum | σ_max | 0.5 | `sigma_max` |
| Warmup steps | N_warm | 5000 | `uncertainty_warmup_steps` |
| LR scale | - | 0.1x | `uncertainty_lr_scale` |

## Key Implementation Details

1. **Log-variance prediction**: MLP outputs `s = log σ`, then `σ = exp(s)` is computed
2. **Clamping**: `σ` is clamped to `[1e-3, 0.5]` after exp() to prevent extreme values
3. **Initialization**: `s_0 = -3` ensures conservative initial uncertainty (σ ≈ 0.05)
4. **Curriculum learning**: Uncertainty disabled for first 5000 steps (configurable)
5. **Learning rate**: Uncertainty MLP uses 0.1x color LR to prevent quick inflation
6. **Deterministic losses**: SDF, eikonal, normal losses remain unchanged (no uncertainty)

## Testing Checklist

- [ ] Verify uncertainty MLP predicts log-variance correctly
- [ ] Verify σ clamping works (check min/max values)
- [ ] Verify warmup stage disables uncertainty correctly
- [ ] Verify learning rate scaling (0.1x) is applied
- [ ] Verify default hyperparameters match ND-SDF recommendations
- [ ] Test with existing configs to ensure backward compatibility

## Notes

- Legacy SSIM-based uncertainty loss is still available if `use_ssim_uncertainty: true`
- All changes are backward compatible (defaults are set appropriately)
- Configuration files can override any default value

