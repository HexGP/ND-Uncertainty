# Result Parameter Values

This document tracks the parameter values for different experimental runs testing heteroscedastic uncertainty affecting RGB L1 loss.

## Results Summary

| Run | Normal C. ↑ | Chamfer ↓ | F-score ↑ | Status |
|-----|-------------|-----------|-----------|--------|
| Baseline (SSIM) | 91.53 | 2.81 | 90.09 | Reference |
| Run 1 | 80.04 | 14.85 | 25.28 | Poor - uncertainty collapsed |
| Run 2 | 83.23 | 8.99 | 45.59 | Improved - fixes applied |
| Run 3 | 86.17 | 5.88 | 64.63 | Good - significant improvement |
| Run 4 | TBD | TBD | TBD | TBD |

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

## Run 3 (Stabilized Uncertainty + Reduced Impact)

**Results:** Normal C. 86.17, Chamfer 5.88, F-score 64.63

**Improvements:** Reduced uncertainty's impact on color learning while stabilizing uncertainty values. Stronger geometry constraints.

**Key Insight:** We're **stabilizing uncertainty values** (preventing collapse/inflation) while **reducing uncertainty's impact** on color learning (less aggressive down-weighting).

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
2. **Stronger regularization (0.15)** - Better prevents σ from inflating/collapsing (stabilizes uncertainty values)
3. **Lower initial uncertainty (-4.0)** - Starts even lower to prevent early inflation (stabilizes initialization)
4. **Stronger geometry constraints** - Better geometry quality improves Normal C. and Chamfer

### Actual Results
- **Normal C.: 86.17** (+2.94 from Run 2, -5.36 from baseline) - Geometry quality improved significantly
- **Chamfer: 5.88** (-3.11 from Run 2, +3.07 from baseline) - Mesh quality much better
- **F-score: 64.63** (+19.04 from Run 2, -25.46 from baseline) - Overall quality improved substantially

### Key Successes
- Uncertainty values are stable (no collapse/inflation)
- Color learning improved (reduced uncertainty dominance worked)
- Geometry quality improved (stronger constraints worked)
- Getting closer to baseline performance

---

## Run 4 (Prioritize Color Learning + Strengthen Geometry)

**Goal:** Further improve color/rendering quality (F-score gap: -25.46) while maintaining geometry improvements.

**Results:** TBD

### Uncertainty Parameters
- `use_uncertainty: true`
- `use_ssim_uncertainty: false` (heteroscedastic mode)
- `weight_unc: 0.3` (λ_color: **reduced from 0.5** - further prioritize color learning, reduce uncertainty dominance)
- `unc_lambda_reg: 0.2` (β: **increased from 0.15** - stronger regularization to maintain stability with lower weight_unc)
- `init_log_sigma: -4.0` (s₀: σ₀ ≈ 0.018, **kept at -4.0** - working well, no change needed)
- `sigma_min: 0.01` (minimum σ clamp for numerical stability)
- `sigma_max: 1000.0` (maximum σ clamp - effectively removed, regularizer prevents unbounded growth)
- `uncertainty_warmup_steps: 5000`
- `uncertainty_lr_scale: 0.1`

### Geometry Parameters
- `lambda_eik: 0.1` (eikonal loss weight - **increased from 0.08** - further strengthen geometry constraint, Normal C. gap: -5.36)
- `lambda_ab_normal: 0.08` (adaptive normal loss weight - **increased from 0.06** - further strengthen normal constraint)
- `lambda_rgb_l1: 1.0` (replaced by heteroscedastic loss)

### Rationale
1. **Further reduced uncertainty weight (0.3)** - F-score gap is still large (-25.46), need to prioritize color learning even more. Uncertainty still affects color but much less aggressively.
2. **Stronger regularization (0.2)** - With lower weight_unc, need stronger regularization to maintain σ stability and prevent inflation.
3. **Kept initial uncertainty (-4.0)** - Working well in Run 3, no need to change.
4. **Further strengthened geometry constraints** - Normal C. gap is -5.36, Chamfer gap is +3.07. Stronger eikonal (0.1) and normal (0.08) constraints should help close these gaps.

### Expected Improvements
- Better color/rendering quality (F-score should improve significantly with weight_unc: 0.3)
- Better geometry quality (Normal C. and Chamfer should improve with stronger constraints)
- Stable uncertainty values (stronger regularization maintains stability)
- Closer to baseline performance across all metrics

---

## Parameter Comparison Table

| Parameter | Run 1 | Run 2 | Run 3 | Run 4 | Baseline (SSIM) |
|-----------|-------|-------|-------|-------|-----------------|
| `weight_unc` | 1.0 | 1.0 | **0.5** | **0.3** | 1.0 |
| `unc_lambda_reg` | 0.05 | 0.1 | **0.15** | **0.2** | 0.5 |
| `init_log_sigma` | -3.0 | -3.5 | **-4.0** | **-4.0** | N/A |
| `sigma_min` | 0.01 | 0.01 | 0.01 | 0.01 | 0.1 |
| `sigma_max` | 0.5 | 1000.0 | 1000.0 | 1000.0 | N/A |
| `lambda_eik` | 0.05 | 0.05 | **0.08** | **0.1** | 0.05 |
| `lambda_ab_normal` | 0.04 | 0.04 | **0.06** | **0.08** | 0.04 |
| `use_ssim_uncertainty` | false | false | false | false | true |

---

## Notes

- **Baseline** uses SSIM-based uncertainty (separate loss, doesn't affect RGB L1) - this is why it performs better
- **Run 1-4** use heteroscedastic uncertainty (affects RGB L1 directly) - more challenging but what professor requested
- **Run 3 Strategy:** 
  - **Stabilizing uncertainty values** (prevent collapse/inflation via stronger regularization and lower initialization)
  - **Reducing uncertainty's impact** (weight_unc: 0.5 makes uncertainty less dominant, allowing better color learning)
  - **Strengthening geometry** (higher lambda_eik and lambda_ab_normal improve geometry quality)
- **Run 4 Strategy:**
  - **Further reduce uncertainty impact** (weight_unc: 0.3 prioritizes color learning to close F-score gap)
  - **Maintain stability** (unc_lambda_reg: 0.2 keeps σ stable with lower weight_unc)
  - **Further strengthen geometry** (lambda_eik: 0.1, lambda_ab_normal: 0.08 to close remaining geometry gaps)
- All runs use `uncertainty_warmup_steps: 5000` to let geometry initialize before uncertainty kicks in
- **Progress:** Run 1 → Run 2 → Run 3 shows consistent improvement across all metrics
- **Run 4 Focus:** Prioritize color learning (F-score) while maintaining geometry improvements
