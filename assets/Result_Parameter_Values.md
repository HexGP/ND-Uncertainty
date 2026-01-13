# Result Parameter Values

This document tracks the parameter values for different experimental runs testing heteroscedastic uncertainty affecting RGB L1 loss.

## Results Summary

| Run | Normal C. ↑ | Chamfer ↓ | F-score ↑ | Clamp | Status |
|-----|-------------|-----------|-----------|-------|--------|
| Baseline (SSIM) | 91.53 | 2.81 | 90.09 | None | Reference |
| Run 1 | 80.04 | 14.85 | 25.28 | 0.5 | Poor - uncertainty collapsed |
| Run 2 | 83.23 | 8.99 | 45.59 | 1000.0 | Improved - fixes applied |
| Run 3 | 86.17 | 5.88 | 64.63 | 1000.0 | Good - significant improvement |
| Run 4 | 86.82 | 6.01 | 66.63 | 1000.0 | Marginal improvement - diminishing returns |
| Run 5: Hybrid | 88.75 | 3.71 | 78.87 | 1000.0 | Excellent - hybrid approach breakthrough |
| Run 6: Tuned Hybrid | 88.95 | 3.88 | 76.71 | 1000.0 | Regression - 0.4 hybrid_weight performed worse than Run 5 |
| Run 7: Reduced Hybrid | 88.00 | 4.38 | 71.59 | 1000.0 | Significant regression - all changes made performance worse |
| Run 8: Restored Run 5 | 87.70 | 3.89 | 78.29 | 1000.0 | Slightly below Run 5 - within variance, confirms approach |
| Run 9: Fine-tuned Hybrid | TBD | TBD | TBD | 1000.0 | In progress - conservative increase in uncertainty (0.32) |
| Run 10: Higher Max Clamp | TBD | TBD | TBD | 100000.0 | In progress - increased sigma_max to 100000.0 to match baseline approach |

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

**Results:** Normal C. 86.82, Chamfer 6.01, F-score 66.63

**Goal:** Further improve color/rendering quality (F-score gap: -25.46) while maintaining geometry improvements.

**Outcome:** Marginal improvements - diminishing returns observed.

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

### Actual Results
- **Normal C.: 86.82** (+0.65 from Run 3, -4.71 from baseline) - Small improvement, still below baseline
- **Chamfer: 6.01** (+0.13 from Run 3, +3.20 from baseline) - **Slightly worse** than Run 3, geometry slightly degraded
- **F-score: 66.63** (+2.00 from Run 3, -23.46 from baseline) - Small improvement, F-score gap still large

### Analysis
**Diminishing Returns:** Run 4 shows much smaller improvements compared to Run 2→Run 3:
- Run 2→Run 3: Normal C. +2.94, Chamfer -3.11, F-score +19.04 (large improvements)
- Run 3→Run 4: Normal C. +0.65, Chamfer +0.13, F-score +2.00 (marginal improvements)

**Trade-off Observed:**
- **F-score improved** (+2.00) - Lower `weight_unc: 0.3` helped color learning
- **Chamfer got worse** (+0.13) - Stronger geometry constraints (`lambda_eik: 0.1`, `lambda_ab_normal: 0.08`) may have over-constrained, or uncertainty reduction affected geometry learning
- **Normal C. improved slightly** (+0.65) - Geometry quality maintained but not significantly improved

**Key Insight:** Further reducing `weight_unc` to 0.3 helped F-score but may have hurt geometry (Chamfer). The heteroscedastic approach may be reaching its limit - uncertainty affecting color directly creates a fundamental trade-off between color and geometry quality.

---

## Parameter Comparison Table

| Parameter | Run 1 | Run 2 | Run 3 | Run 4 | Run 5: Hybrid | Run 6 | Run 7 | Baseline (SSIM) |
|-----------|-------|-------|-------|-------|---------------|-------|-------|-----------------|
| `hybrid_weight` | N/A | N/A | N/A | N/A | **0.3** | **0.4** | **0.25** | **0.3** | **0.32** | **0.32** | N/A |
| `weight_unc` | 1.0 | 1.0 | **0.5** | **0.3** | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `unc_lambda_reg` | 0.05 | 0.1 | **0.15** | **0.2** | **0.15** | **0.18** | **0.12** | **0.15** | **0.16** | **0.16** | 0.5 |
| `init_log_sigma` | -3.0 | -3.5 | **-4.0** | **-4.0** | **-4.0** | **-4.0** | **-3.5** | **-4.0** | **-4.0** | **-4.0** | N/A |
| `sigma_min` | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | **0.015** | **0.01** | **0.01** | **0.01** | 0.1 |
| `sigma_max` | 0.5 | 1000.0 | 1000.0 | 1000.0 | 1000.0 | 1000.0 | 1000.0 | 1000.0 | 1000.0 | **100000.0** | N/A |
| `lambda_eik` | 0.05 | 0.05 | **0.08** | **0.1** | **0.08** | **0.09** | **0.08** | **0.08** | **0.08** | 0.05 |
| `lambda_ab_normal` | 0.04 | 0.04 | **0.06** | **0.08** | **0.06** | **0.07** | **0.06** | **0.06** | **0.06** | 0.04 |
| `use_ssim_uncertainty` | false | false | false | false | false | false | false | false | false | true |

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
- **Run 4 Outcome:** Marginal improvements with diminishing returns. Trade-off observed: F-score improved but Chamfer got slightly worse. Heteroscedastic approach may be reaching its limit.

---

## Run 5: Hybrid Approach (Breakthrough)

**Results:** Normal C. 88.75, Chamfer 3.71, F-score 78.87

**Breakthrough:** Hybrid approach (30% heteroscedastic + 70% standard RGB L1) achieved significant improvements across all metrics, getting much closer to baseline.

### Uncertainty Parameters
- `use_uncertainty: true`
- `use_ssim_uncertainty: false` (heteroscedastic mode)
- `hybrid_weight: 0.3` (**NEW**: 30% heteroscedastic, 70% standard RGB L1 blend)
- `weight_unc: 1.0` (λ_color: weight for heteroscedastic component in blend)
- `unc_lambda_reg: 0.15` (β: regularization weight, reduced from 0.2 since hybrid is more stable)
- `init_log_sigma: -4.0` (s₀: σ₀ ≈ 0.018, kept from Run 3/4)
- `sigma_min: 0.01` (minimum σ clamp for numerical stability)
- `sigma_max: 1000.0` (effectively no max clamp, regularizer prevents unbounded growth)
- `uncertainty_warmup_steps: 5000`
- `uncertainty_lr_scale: 0.1`

### Geometry Parameters
- `lambda_eik: 0.08` (eikonal loss weight - using Run 3 value, good balance)
- `lambda_ab_normal: 0.06` (adaptive normal loss weight - using Run 3 value, good balance)
- `lambda_rgb_l1: 1.0` (used in hybrid blend, not replaced)

### Rationale
1. **Hybrid approach (0.3 blend)** - Mixes heteroscedastic uncertainty (30%) with standard RGB L1 (70%)
   - Maintains strong color learning from standard RGB L1 (70% weight)
   - Adds uncertainty awareness without fully replacing color learning (30% weight)
   - Better balance than pure heteroscedastic replacement
2. **Reduced regularization (0.15)** - Hybrid approach is more stable, less aggressive regularization needed
3. **Geometry constraints (Run 3 values)** - Good balance from Run 3, no need to over-constrain

### Actual Results
- **Normal C.: 88.75** (+1.93 from Run 4, -2.78 from baseline) - **Significant improvement**, very close to baseline
- **Chamfer: 3.71** (-2.30 from Run 4, +0.90 from baseline) - **Major improvement**, almost matches baseline
- **F-score: 78.87** (+12.24 from Run 4, -11.22 from baseline) - **Breakthrough improvement**, much closer to baseline

### Key Successes
- **Hybrid approach works!** - Mixing heteroscedastic with standard RGB L1 achieves much better balance
- **All metrics improved significantly** - No trade-offs, consistent improvement across Normal C., Chamfer, and F-score
- **Much closer to baseline** - Normal C. gap: -2.78 (vs -4.71 in Run 4), Chamfer gap: +0.90 (vs +3.20 in Run 4), F-score gap: -11.22 (vs -23.46 in Run 4)
- **Best results so far** - Hybrid approach proves to be the right direction

### Visualization Issue
- **Uncertainty images appear uniform (purple/blue) at epoch 2400** - Uncertainty values have converged to similar high values, causing uniform visualization
- **Variation visible at epoch 720** - Earlier in training, uncertainty values had more variation
- **Fix needed**: Use percentile-based normalization instead of min/max to show variation even when values are saturated

---

## Run 6: Tuned Hybrid Approach (Target: Exceed Baseline)

**Goal:** Close remaining gaps to baseline (Normal C. -2.78, Chamfer +0.90, F-score -11.22) by fine-tuning hybrid approach.

**Strategy:** 
1. **Increase hybrid_weight (0.3 → 0.4)** - Get more uncertainty benefit (40% heteroscedastic, 60% RGB L1) while maintaining strong color learning
2. **Strengthen regularization (0.15 → 0.18)** - Maintain stability with higher hybrid_weight
3. **Slightly strengthen geometry (eik: 0.08→0.09, normal: 0.06→0.07)** - Close Normal C. gap (-2.78)

### Uncertainty Parameters
- `use_uncertainty: true`
- `use_ssim_uncertainty: false` (heteroscedastic mode)
- `hybrid_weight: 0.4` (**INCREASED from 0.3** - 40% heteroscedastic, 60% standard RGB L1)
- `weight_unc: 1.0` (λ_color: weight for heteroscedastic component in blend)
- `unc_lambda_reg: 0.18` (**INCREASED from 0.15** - stronger regularization to maintain stability)
- `init_log_sigma: -4.0` (s₀: σ₀ ≈ 0.018, kept from Run 5)
- `sigma_min: 0.01` (minimum σ clamp for numerical stability)
- `sigma_max: 1000.0` (effectively no max clamp, regularizer prevents unbounded growth)
- `uncertainty_warmup_steps: 5000`
- `uncertainty_lr_scale: 0.1`

### Geometry Parameters
- `lambda_eik: 0.09` (**INCREASED from 0.08** - slightly stronger geometry constraint to close Normal C. gap)
- `lambda_ab_normal: 0.07` (**INCREASED from 0.06** - slightly stronger normal constraint)
- `lambda_rgb_l1: 1.0` (used in hybrid blend, not replaced)

### Rationale
1. **Increased hybrid_weight (0.4)** - More uncertainty benefit (40% vs 30%) while keeping 60% standard RGB L1 for strong color learning
   - Should help F-score (color quality) by better uncertainty-aware learning
   - Still maintains majority RGB L1 (60%) to prevent color degradation
2. **Stronger regularization (0.18)** - With higher hybrid_weight, need slightly more regularization to prevent uncertainty inflation
3. **Slightly stronger geometry (0.09, 0.07)** - Small increases to close Normal C. gap (-2.78) without over-constraining

### Expected Results
- **Target Normal C.:** > 90.0 (close -2.78 gap)
- **Target Chamfer:** < 3.0 (close +0.90 gap)
- **Target F-score:** > 85.0 (close -11.22 gap, primary focus)

### Key Hypothesis
- **More uncertainty benefit (40% vs 30%)** will improve F-score by better handling uncertain regions
- **Maintained RGB L1 strength (60%)** will prevent color degradation
- **Slightly stronger geometry** will close Normal C. gap
- **Stronger regularization** will maintain stability

### Actual Results
- **Normal C.: 88.95** (+0.20 from Run 5, -2.58 from baseline) - **Slight improvement** but still below baseline
- **Chamfer: 3.88** (+0.17 from Run 5, +1.07 from baseline) - **Worse than Run 5**, moved away from baseline
- **F-score: 76.71** (-2.16 from Run 5, -13.38 from baseline) - **Regression**, primary target failed

### Analysis
- **Run 6 performed worse than Run 5** - Increasing `hybrid_weight` from 0.3 to 0.4 hurt performance
- **F-score decreased** - More uncertainty (40%) interfered with color learning instead of helping
- **Chamfer increased** - Geometry quality degraded slightly
- **Key Insight**: 30% heteroscedastic (Run 5) appears to be the optimal hybrid weight. More uncertainty doesn't help.

### Visualization Issue
- **Uncertainty heatmaps appear lime green (uniform)** - Uncertainty values are high and uniform, suggesting saturation or convergence to high values
- **Regularization may need to be stronger** to pull uncertainty down from high values

---

## Run 7: Reduced Hybrid + Prevent Collapse (Target: Maintain Variation + Improve F-score)

**Goal:** Prevent uncertainty collapse (all values hitting sigma_min) while improving F-score by reducing hybrid_weight and weakening regularization.

**Strategy:** 
1. **Reduce hybrid_weight (0.4 → 0.25)** - Try less uncertainty (25% heteroscedastic, 75% RGB L1) since 0.4 performed worse than 0.3
2. **Weaken regularization (0.2 → 0.12)** - Prevent uncertainty collapse by allowing more variation (weaker regularization)
3. **Increase initial uncertainty (-4.5 → -3.5)** - Start with more uncertainty to prevent early collapse
4. **Increase sigma_min (0.01 → 0.015)** - Give more room before hitting clamp boundary
5. **Revert geometry constraints (0.09→0.08, 0.07→0.06)** - Back to Run 5 values since Run 6's increases didn't help

### Uncertainty Parameters
- `use_uncertainty: true`
- `use_ssim_uncertainty: false` (heteroscedastic mode)
- `hybrid_weight: 0.25` (**REDUCED from 0.4** - 25% heteroscedastic, 75% standard RGB L1, try less uncertainty)
- `weight_unc: 1.0` (λ_color: weight for heteroscedastic component in blend)
- `unc_lambda_reg: 0.12` (**REDUCED from 0.2 to 0.12** - weaker regularization to prevent collapse, allow more variation)
- `init_log_sigma: -3.5` (**INCREASED from -4.5 to -3.5** - σ₀ ≈ 0.03, start with more uncertainty to prevent early collapse)
- `sigma_min: 0.015` (**INCREASED from 0.01 to 0.015** - give more room before hitting clamp, prevent early collapse)
- `sigma_max: 1000.0` (effectively no max clamp, regularizer prevents unbounded growth)
- `uncertainty_warmup_steps: 5000`
- `uncertainty_lr_scale: 0.1`

### Geometry Parameters
- `lambda_eik: 0.08` (**REVERTED from 0.09** - back to Run 5 value, Run 6's increase didn't help)
- `lambda_ab_normal: 0.06` (**REVERTED from 0.07** - back to Run 5 value, Run 6's increase didn't help)
- `lambda_rgb_l1: 1.0` (used in hybrid blend, not replaced)

### Rationale
1. **Reduced hybrid_weight (0.25)** - Since 0.4 performed worse than 0.3, try even less uncertainty (25% vs 30%)
   - More RGB L1 (75%) should help color learning and F-score
   - Less uncertainty interference should improve overall performance
2. **Weaker regularization (0.12)** - Prevent uncertainty collapse by allowing more variation
   - Previous runs showed collapse to sigma_min (0.01) at epoch 2400
   - Weaker regularization (0.12 vs 0.2) will allow uncertainty to maintain spatial variation
   - Still strong enough to prevent unbounded inflation (sigma_max: 1000.0)
3. **Higher initial uncertainty (-3.5)** - Start with more uncertainty to prevent early collapse
   - Previous runs started at -4.0 to -4.5 and collapsed quickly
   - Starting at -3.5 (σ₀ ≈ 0.03) gives more headroom before hitting sigma_min
4. **Higher sigma_min (0.015)** - Give more room before hitting clamp boundary
   - Previous runs collapsed to 0.01 (sigma_min), showing uniform values
   - Increasing to 0.015 provides more room for variation
5. **Reverted geometry constraints** - Run 6's increases didn't help, back to Run 5's proven values

### Expected Results
- **Target Normal C.:** > 88.75 (match or exceed Run 5)
- **Target Chamfer:** < 3.71 (match or exceed Run 5)
- **Target F-score:** > 78.87 (match or exceed Run 5, primary focus)
- **Target Visualization:** Maintain variation in uncertainty heatmaps at epoch 2400 (not uniform lime green/collapsed)

### Key Hypothesis
- **Less uncertainty (25% vs 30-40%)** will improve F-score by reducing interference with color learning
- **Weaker regularization (0.12)** will prevent collapse and maintain spatial variation in uncertainty
- **Higher initial uncertainty (-3.5)** will prevent early collapse
- **Higher sigma_min (0.015)** will give more room before hitting clamp
- **Reverted geometry** will maintain Run 5's proven balance

### Actual Results
- **Normal C.: 88.00** (-0.75 from Run 5, -0.95 from Run 6, -3.53 from baseline) - **Regression**, worse than both Run 5 and Run 6
- **Chamfer: 4.38** (+0.67 from Run 5, +0.50 from Run 6, +1.57 from baseline) - **Worse**, geometry degraded
- **F-score: 71.59** (-7.28 from Run 5, -5.12 from Run 6, -18.50 from baseline) - **Significant regression**, biggest drop

### Analysis
- **Run 7 performed worse than both Run 5 and Run 6** - All parameter changes made performance worse
- **F-score dropped significantly** (-7.28 from Run 5) - Primary failure, too little uncertainty benefit
- **Chamfer increased** (+0.67 from Run 5) - Geometry quality degraded
- **Variation window shifted earlier** (240→480 vs 240→720) - Uncertainty collapsed sooner despite trying to prevent it
- **Key Insight**: All anti-collapse changes backfired - reducing hybrid_weight and weakening regularization too much hurt performance

### Visualization Issue
- **Variation window shifted earlier** (240→480 instead of 240→720) - Uncertainty collapsed earlier despite changes intended to prevent it
- **Faster collapse** confirms that Run 7's parameters were too aggressive

---

## Run 8: Restored Run 5 Parameters (Back to Best Configuration)

**Goal:** Restore Run 5's proven parameters after Run 7's regression. Use Run 5 as baseline for future fine-tuning.

**Strategy:** 
1. **Restore all Run 5 parameters** - Return to proven configuration that achieved best results
2. **Use as stable baseline** - Establish Run 5 as reference point for future experiments

### Uncertainty Parameters
- `use_uncertainty: true`
- `use_ssim_uncertainty: false` (heteroscedastic mode)
- `hybrid_weight: 0.3` (**RESTORED from Run 5** - 30% heteroscedastic, 70% standard RGB L1)
- `weight_unc: 1.0` (λ_color: weight for heteroscedastic component in blend)
- `unc_lambda_reg: 0.15` (**RESTORED from Run 5** - moderate regularization)
- `init_log_sigma: -4.0` (**RESTORED from Run 5** - σ₀ ≈ 0.018, proven to work well)
- `sigma_min: 0.01` (**RESTORED from Run 5** - lower clamp gives more room)
- `sigma_max: 1000.0` (effectively no max clamp, regularizer prevents unbounded growth)
- `uncertainty_warmup_steps: 5000`
- `uncertainty_lr_scale: 0.1`

### Geometry Parameters
- `lambda_eik: 0.08` (**RESTORED from Run 5** - proven geometry constraint)
- `lambda_ab_normal: 0.06` (**RESTORED from Run 5** - proven normal constraint)
- `lambda_rgb_l1: 1.0` (used in hybrid blend, not replaced)

### Rationale
1. **Restored Run 5 parameters** - Return to proven configuration that achieved best results so far
2. **Run 7 confirmed Run 5 is better** - All changes in Run 7 made performance worse
3. **Establish stable baseline** - Use Run 5 as reference point for future fine-tuning
4. **Proven balance** - Run 5 achieved best hybrid approach results (88.75, 3.71, 78.87)

### Expected Results
- **Target Normal C.:** ≈ 88.75 (match Run 5)
- **Target Chamfer:** ≈ 3.71 (match Run 5)
- **Target F-score:** ≈ 78.87 (match Run 5)
- **Target Visualization:** Variation window around 240→720 (similar to Run 5)

### Key Hypothesis
- **Restored parameters will match Run 5** - Same configuration should produce similar results
- **Run 5 is optimal hybrid configuration** - 30% heteroscedastic, 70% RGB L1 with moderate regularization
- **This establishes stable baseline** - Future experiments can tune from this proven point

### Actual Results
- **Normal C.: 87.70** (-1.05 from Run 5, -3.83 from baseline) - **Slightly worse**, geometry slightly degraded
- **Chamfer: 3.89** (+0.18 from Run 5, +1.08 from baseline) - **Slightly worse**, within variance
- **F-score: 78.29** (-0.58 from Run 5, -11.80 from baseline) - **Close to Run 5**, within variance

### Analysis
- **Run 8 did not exactly match Run 5** - All metrics slightly worse, but within reasonable variance
- **F-score is very close** (-0.58) - Suggests Run 5's approach is reproducible
- **Normal C. dropped more** (-1.05) - May indicate slight geometry degradation or variance
- **Still validates Run 5** - Confirms Run 5's parameters are the best hybrid configuration
- **Possible reasons for difference**: Code fixes (eikonal loss fallbacks), training variance, or environment differences

---

## Run 9: Fine-tuned Hybrid (Conservative F-score Improvement)

**Goal:** Improve F-score with conservative parameter tweaks while maintaining geometry quality.

**Strategy:** 
1. **Slightly increase hybrid_weight (0.3 → 0.32)** - Conservative increase (between Run 5's 0.3 and Run 6's 0.4) to get more uncertainty benefit
2. **Slightly strengthen regularization (0.15 → 0.16)** - Maintain stability with higher hybrid_weight
3. **Keep all other parameters** - Maintain Run 5's proven geometry constraints and initialization

### Uncertainty Parameters
- `use_uncertainty: true`
- `use_ssim_uncertainty: false` (heteroscedastic mode)
- `hybrid_weight: 0.32` (**INCREASED from 0.3 to 0.32** - 32% heteroscedastic, 68% standard RGB L1, conservative increase)
- `weight_unc: 1.0` (λ_color: weight for heteroscedastic component in blend)
- `unc_lambda_reg: 0.16` (**INCREASED from 0.15 to 0.16** - slightly stronger regularization to maintain stability)
- `init_log_sigma: -4.0` (**KEPT from Run 5** - σ₀ ≈ 0.018, proven to work well)
- `sigma_min: 0.01` (**KEPT from Run 5** - lower clamp gives more room)
- `sigma_max: 1000.0` (effectively no max clamp, regularizer prevents unbounded growth)
- `uncertainty_warmup_steps: 5000`
- `uncertainty_lr_scale: 0.1`

### Geometry Parameters
- `lambda_eik: 0.08` (**KEPT from Run 5** - proven geometry constraint)
- `lambda_ab_normal: 0.06` (**KEPT from Run 5** - proven normal constraint)
- `lambda_rgb_l1: 1.0` (used in hybrid blend, not replaced)

### Rationale
1. **Conservative increase in hybrid_weight (0.32)** - Between Run 5's 0.3 (best) and Run 6's 0.4 (worse)
   - Should provide more uncertainty benefit for F-score improvement
   - Still maintains majority RGB L1 (68%) to prevent color degradation
   - More conservative than Run 6's 0.4 that hurt F-score
2. **Slightly stronger regularization (0.16)** - Maintain stability with higher hybrid_weight
   - Prevents uncertainty from collapsing or inflating
   - Still moderate (not as strong as Run 4's 0.2)
3. **Keep proven parameters** - Maintain Run 5's geometry constraints and initialization that worked well

### Expected Results
- **Target Normal C.:** ≥ 87.70 (maintain or improve from Run 8)
- **Target Chamfer:** ≤ 3.89 (maintain or improve from Run 8)
- **Target F-score:** > 78.29 (improve from Run 8, ideally > 78.87 to match/exceed Run 5)

### Key Hypothesis
- **Conservative increase (0.32 vs 0.3)** will improve F-score without hurting geometry (unlike Run 6's 0.4)
- **Slightly stronger regularization (0.16)** will maintain stability
- **Proven geometry constraints** will maintain Normal C. and Chamfer quality
- **This should be the sweet spot** between Run 5 (0.3) and Run 6 (0.4)

---

## Run 10: Higher Max Clamp (Match Baseline Approach)

**Goal:** Increase `sigma_max` to 100,000 to match baseline approach (effectively no max clamp) while keeping `sigma_min` for numerical stability.

**Strategy:** 
1. **Increase sigma_max (1000.0 → 100000.0)** - Much higher clamp to effectively match baseline (no max clamp)
2. **Keep all Run 9 parameters** - Maintain proven hybrid approach configuration

### Uncertainty Parameters
- `use_uncertainty: true`
- `use_ssim_uncertainty: false` (heteroscedastic mode)
- `hybrid_weight: 0.32` (**KEPT from Run 9** - 32% heteroscedastic, 68% standard RGB L1)
- `weight_unc: 1.0` (λ_color: weight for heteroscedastic component in blend)
- `unc_lambda_reg: 0.16` (**KEPT from Run 9** - slightly stronger regularization)
- `init_log_sigma: -4.0` (**KEPT from Run 9** - σ₀ ≈ 0.018, proven to work well)
- `sigma_min: 0.01` (**KEPT from Run 9** - needed for numerical stability)
- `sigma_max: 100000.0` (**INCREASED from 1000.0 to 100000.0** - much higher clamp, effectively no constraint like baseline)
- `uncertainty_warmup_steps: 5000`
- `uncertainty_lr_scale: 0.1`

### Geometry Parameters
- `lambda_eik: 0.08` (**KEPT from Run 9** - proven geometry constraint)
- `lambda_ab_normal: 0.06` (**KEPT from Run 9** - proven normal constraint)
- `lambda_rgb_l1: 1.0` (used in hybrid blend, not replaced)

### Rationale
1. **Much higher sigma_max (100000.0)** - Effectively no max clamp, matching baseline approach
   - Baseline doesn't use `sigma_max` for SSIM-based uncertainty
   - Regularizer should prevent unbounded growth (same as with 1000.0)
   - Provides safety net (unlike removing clamp entirely) but unlikely to be hit
   - Easier to debug (can see if values approach this bound)
2. **Keep all other Run 9 parameters** - Maintain proven hybrid configuration
   - Run 9's conservative tweaks (0.32 hybrid_weight, 0.16 regularization) are good
   - Only testing if higher max clamp affects uncertainty behavior

### Expected Results
- **Target Normal C.:** ≥ 87.70 (maintain or improve from Run 8/9)
- **Target Chamfer:** ≤ 3.89 (maintain or improve from Run 8/9)
- **Target F-score:** > 78.29 (improve from Run 8, ideally > 78.87 to match/exceed Run 5)

### Key Hypothesis
- **Higher sigma_max (100000.0 vs 1000.0)** should match baseline approach more closely
- **No practical difference** - Both are so high they're effectively no constraint
- **Regularizer still prevents unbounded growth** - Same protection as before
- **If Run 9 performs well, Run 10 should match** - Only change is max clamp value

---
