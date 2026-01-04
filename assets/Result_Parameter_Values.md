# Result Parameter Values

This document tracks the parameter values for different experimental runs testing heteroscedastic uncertainty affecting RGB L1 loss.

## Results Summary

| Run | Normal C. ↑ | Chamfer ↓ | F-score ↑ | Status |
|-----|-------------|-----------|-----------|--------|
| Baseline (SSIM) | 91.53 | 2.81 | 90.09 | Reference |
| Run 1 | 80.04 | 14.85 | 25.28 | Poor - uncertainty collapsed |
| Run 2 | 83.23 | 8.99 | 45.59 | Improved - fixes applied |
| Run 3 | 86.17 | 5.88 | 64.63 | Good - significant improvement |
| Run 4 | 86.82 | 6.01 | 66.63 | Marginal improvement - diminishing returns |
| Run 5: Hybrid | 88.75 | 3.71 | 78.87 | Excellent - hybrid approach breakthrough |
| Run 6: Tuned Hybrid | TBD | TBD | TBD | In progress - increased hybrid_weight to 0.4 |

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

| Parameter | Run 1 | Run 2 | Run 3 | Run 4 | Run 5: Hybrid | Baseline (SSIM) |
|-----------|-------|-------|-------|-------|---------------|-----------------|
| `hybrid_weight` | N/A | N/A | N/A | N/A | **0.3** | N/A |
| `weight_unc` | 1.0 | 1.0 | **0.5** | **0.3** | 1.0 | 1.0 |
| `unc_lambda_reg` | 0.05 | 0.1 | **0.15** | **0.2** | **0.15** | 0.5 |
| `init_log_sigma` | -3.0 | -3.5 | **-4.0** | **-4.0** | **-4.0** | N/A |
| `sigma_min` | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.1 |
| `sigma_max` | 0.5 | 1000.0 | 1000.0 | 1000.0 | 1000.0 | N/A |
| `lambda_eik` | 0.05 | 0.05 | **0.08** | **0.1** | **0.08** | 0.05 |
| `lambda_ab_normal` | 0.04 | 0.04 | **0.06** | **0.08** | **0.06** | 0.04 |
| `use_ssim_uncertainty` | false | false | false | false | false | true |

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

---
