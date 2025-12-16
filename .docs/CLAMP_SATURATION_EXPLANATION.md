# Clamp Saturation Monitoring - Simple Explanation

## What is "Clamp Saturation"?

**Clamp saturation** means that sigma (uncertainty) values are hitting the **clamp boundaries** (min or max limits).

### Example:
- Your sigma values are clamped to `[1e-3, 0.5]` (min = 0.001, max = 0.5)
- If many sigma values are **exactly 0.5**, they're "saturated at max"
- If many sigma values are **exactly 0.001**, they're "saturated at min"

## Why This is a Problem

### When Sigma Hits Max (0.5):
1. **All values become the same** → uniform uncertainty → uniform visualization (lime green/blue)
2. **Gradients stop flowing** → sigma stops learning → can't distinguish high/low uncertainty
3. **Loss becomes ineffective** → heteroscedastic loss can't down-weight uncertain regions

### When Sigma Hits Min (0.001):
1. **Over-emphasizes certain predictions** → can cause overfitting
2. **Loss becomes too large** → `(1/(2*0.001²)) * error² = 500,000 * error²` (huge weight)

## What the Monitoring Does

The code I added does this:

```python
# Count how many sigma values are at clamp bounds
pct_at_min = (beta <= sigma_min + 1e-5).mean() * 100  # % at min
pct_at_max = (beta >= sigma_max - 1e-5).mean() * 100  # % at max

# Log to TensorBoard
loger.add_scalar('uncertainty/pct_at_max_clamp', pct_at_max, step)

# Print warning if >50% are saturated
if pct_at_max > 50:
    print(f"[WARNING] {pct_at_max:.1f}% of sigma at max clamp. Sigma is inflating!")
```

## What You'll See

### In Console (during training):
```
[WARNING] Step 5000: 75.3% of sigma at max clamp (0.5). 
Sigma is inflating - check learning rate and regularization.
```

### In TensorBoard:
- **`uncertainty/pct_at_max_clamp`**: Percentage of values at max (0.5)
- **`uncertainty/pct_at_min_clamp`**: Percentage of values at min (0.001)

**Good values**: < 10% at either bound (sigma is learning properly)
**Bad values**: > 50% at either bound (sigma is saturated, needs fixing)

## How to Use This Information

### If `pct_at_max_clamp > 50%`:
- **Problem**: Sigma is inflating too quickly
- **Fixes**:
  1. Reduce `uncertainty_lr_scale` from 0.1 to 0.05
  2. Increase `variance_weight` (regularization) from 0.1 to 0.2
  3. Check if warmup is working (should be 0% during warmup)

### If `pct_at_min_clamp > 50%`:
- **Problem**: Sigma is collapsing to minimum
- **Fixes**:
  1. Check `init_log_sigma` is -3.0 (not too low)
  2. Reduce regularization weight
  3. Check if color loss is too strong

## Example: What Happened in Your Case

From your diagnostic output:
- **All uncertainty images are uniform** (1 unique color)
- **This means**: ~100% of sigma values are at the same clamp bound
- **Likely cause**: Sigma inflated to max (0.5) and stayed there

**With monitoring**, you would have seen:
```
[WARNING] Step 1000: 85.2% of sigma at max clamp (0.5). Sigma is inflating!
[WARNING] Step 2000: 92.1% of sigma at max clamp (0.5). Sigma is inflating!
[WARNING] Step 3000: 98.5% of sigma at max clamp (0.5). Sigma is inflating!
```

This would have alerted you **early** that sigma was inflating, allowing you to fix it before training completed.

## Summary

**Clamp saturation monitoring** = Tracking how many sigma values hit the min/max bounds

**Why it matters** = Helps catch training issues early (sigma inflating/collapsing)

**What to do** = Check TensorBoard logs, if >50% saturated, adjust learning rate or regularization
