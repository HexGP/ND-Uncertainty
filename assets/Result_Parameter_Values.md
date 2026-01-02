# Result Parameter Values

This document tracks the parameter values for different experimental runs testing heteroscedastic uncertainty affecting RGB L1 loss.

## Results Summary

| Run | Normal C. ↑ | Chamfer ↓ | F-score ↑ | Status |
|-----|-------------|-----------|-----------|--------|
| Baseline (SSIM) | 91.53 | 2.81 | 90.09 | Reference |
| Run 1 | 80.04 | 14.85 | 25.28 | Poor - uncertainty collapsed |
| Run 2 | 83.23 | 8.99 | 45.59 | Improved - fixes applied |
| Run 3 | TBD | TBD | TBD | Suggested improvements |

---

## Run 1 (Initial Heteroscedastic Attempt)

**Results:** Normal C. 80.04, Chamfer 14.85, F-score 25.28

**Issues:** Uncertainty values collapsed/inflated, uniform uncertainty images, poor color learning.

### Uncertainty Parameters
- `use_uncertainty: true`
- `use_ssim_uncertainty: false` (heteroscedastic mode)
- `weight_unc: 1.0` (λ_color: full weight on heteroscedastic color loss)
- `unc_lambda_reg: 0.05` (β: weak regularization, allowed σ inflation)
- `init_log_sigma: -3.0` (s₀: σ₀ ≈ 0.05, higher initial uncertainty)
- `sigma_min: 0.01` (minimum σ clamp for numerical stability)
- `sigma_max: 0.5` (maximum σ clamp - too restrictive, caused saturation)
- `uncertainty_warmup_steps: 5000`
- `uncertainty_lr_scale: 0.1`

### Geometry Parameters
- `lambda_eik: 0.05` (eikonal loss weight)
- `lambda_ab_normal: 0.04` (adaptive normal loss weight)
- `lambda_rgb_l1: 1.0` (replaced by heteroscedastic loss)

### Key Problems
1. **Sigma max clamp too restrictive (0.5)** - All uncertainty values hit clamp boundary
2. **Weak regularization (0.05)** - Couldn't prevent σ inflation
3. **Higher initial uncertainty (-3.0)** - Started too high, inflated quickly

---

## Run 2 (Recent Run - Improved)

**Results:** Normal C. 83.23, Chamfer 8.99, F-score 45.59

**Improvements:** Removed restrictive max clamp, increased regularization, lower initial uncertainty.

### Uncertainty Parameters
- `use_uncertainty: true`
- `use_ssim_uncertainty: false` (heteroscedastic mode)
- `weight_unc: 1.0` (λ_color: full weight on heteroscedastic color loss)
- `unc_lambda_reg: 0.1` (β: **increased from 0.05** - stronger regularization)
- `init_log_sigma: -3.5` (s₀: σ₀ ≈ 0.03, **lower from -3.0** - prevents early inflation)
- `sigma_min: 0.01` (minimum σ clamp for numerical stability)
- `sigma_max: 1000.0` (maximum σ clamp - **effectively removed** - allows natural growth)
- `uncertainty_warmup_steps: 5000`
- `uncertainty_lr_scale: 0.1`

### Geometry Parameters
- `lambda_eik: 0.05` (eikonal loss weight)
- `lambda_ab_normal: 0.04` (adaptive normal loss weight)
- `lambda_rgb_l1: 1.0` (replaced by heteroscedastic loss)

### Key Improvements
1. **Removed max clamp (1000.0)** - Uncertainty can grow naturally, regularizer prevents unbounded growth
2. **Stronger regularization (0.1)** - Better prevents σ inflation
3. **Lower initial uncertainty (-3.5)** - Starts lower, prevents early collapse

### Remaining Issues
- Still below baseline performance
- Uncertainty weight (1.0) may be too aggressive, down-weighting color learning too much
- Geometry constraints could be stronger

---

## Run 3 (Suggested Improvements)

**Goal:** Further improve results by reducing uncertainty dominance and strengthening geometry constraints.

### Uncertainty Parameters
- `use_uncertainty: true`
- `use_ssim_uncertainty: false` (heteroscedastic mode)
- `weight_unc: 0.5` (λ_color: **reduced from 1.0** - make uncertainty less dominant, allow stronger color learning)
- `unc_lambda_reg: 0.15` (β: **increased from 0.1** - even stronger regularization to prevent inflation)
- `init_log_sigma: -4.0` (s₀: σ₀ ≈ 0.018, **lower from -3.5** - start even lower to prevent early inflation)
- `sigma_min: 0.01` (minimum σ clamp for numerical stability)
- `sigma_max: 1000.0` (maximum σ clamp - effectively removed, regularizer prevents unbounded growth)
- `uncertainty_warmup_steps: 5000`
- `uncertainty_lr_scale: 0.1`

### Geometry Parameters
- `lambda_eik: 0.08` (eikonal loss weight - **increased from 0.05** - stronger geometry constraint)
- `lambda_ab_normal: 0.06` (adaptive normal loss weight - **increased from 0.04** - stronger normal constraint)
- `lambda_rgb_l1: 1.0` (replaced by heteroscedastic loss)

### Rationale
1. **Reduced uncertainty weight (0.5)** - Uncertainty still affects color but less aggressively, allowing better color learning
2. **Stronger regularization (0.15)** - Better prevents σ from inflating
3. **Lower initial uncertainty (-4.0)** - Starts even lower to prevent early inflation
4. **Stronger geometry constraints** - Better geometry quality should improve Normal C. and Chamfer

### Expected Improvements
- Better color learning (reduced uncertainty dominance)
- More stable uncertainty values (stronger regularization)
- Better geometry quality (stronger constraints)
- Closer to baseline performance

---

## Parameter Comparison Table

| Parameter | Run 1 | Run 2 | Run 3 | Baseline (SSIM) |
|-----------|-------|-------|-------|-----------------|
| `weight_unc` | 1.0 | 1.0 | **0.5** | 1.0 |
| `unc_lambda_reg` | 0.05 | 0.1 | **0.15** | 0.5 |
| `init_log_sigma` | -3.0 | -3.5 | **-4.0** | N/A |
| `sigma_min` | 0.01 | 0.01 | 0.01 | 0.1 |
| `sigma_max` | 0.5 | 1000.0 | 1000.0 | N/A |
| `lambda_eik` | 0.05 | 0.05 | **0.08** | 0.05 |
| `lambda_ab_normal` | 0.04 | 0.04 | **0.06** | 0.04 |
| `use_ssim_uncertainty` | false | false | false | true |

---

## Notes

- **Baseline** uses SSIM-based uncertainty (separate loss, doesn't affect RGB L1) - this is why it performs better
- **Run 1-3** use heteroscedastic uncertainty (affects RGB L1 directly) - more challenging but what professor requested
- **Run 3** attempts to balance uncertainty's effect on color while maintaining geometry quality
- All runs use `uncertainty_warmup_steps: 5000` to let geometry initialize before uncertainty kicks in
