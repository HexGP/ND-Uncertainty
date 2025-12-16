# Uncertainty Implementation Guide: Fixing Clamping & Visualization Issues

## Current Problems Identified

### 1. **Uniform Uncertainty Visualization (Lime Green/Blue/Black)**
- **Symptom**: All uncertainty images show 1 unique color (RGB std = 0.00)
- **Root Cause**: All sigma values are clamped to `sigma_max=0.5`, making them uniform
- **Why This Happens**: Sigma is inflating too quickly during training, hitting the clamp boundary

### 2. **Scan 3 Training Collapse**
- **Symptom**: Only `mesh_240.ply` is valid (25MB), all later meshes are 221-byte empty Scenes
- **Root Cause**: Training collapsed after epoch 240, likely due to sigma inflation causing color loss to vanish

---

## How Clamping Affects Loss & Visualization

### **Loss Calculation Impact**

**Current Implementation** (from `loss_wrapper.py` line 232):
```python
sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)  # [1e-3, 0.5]
L_color = heteroscedastic_color_loss(rgb_pred, rgb_gt, sigma, mask=mask)
```

**Heteroscedastic Loss Formula** (from professor's notes):
```
L_color(r) = (1/(2σ²)) * ||C - Ĉ||² + (1/2) * log(σ²)
```

**What Happens When Clamped:**
1. **If σ → σ_max (0.5)**: 
   - First term: `(1/(2*0.5²)) * error² = 2 * error²` (constant weight)
   - Loss becomes independent of actual uncertainty → model can't learn proper σ
   - **Gradient cutoff**: When σ hits clamp, gradients stop flowing → σ stops learning

2. **If σ → σ_min (1e-3)**:
   - First term: `(1/(2*(1e-3)²)) * error² = 500,000 * error²` (huge weight)
   - Over-emphasizes certain predictions → can cause overfitting

**The Problem**: When all σ values hit `σ_max`, they become uniform → loss can't distinguish between high/low uncertainty regions → visualization shows uniform color.

### **Visualization Impact**

**Current Visualization** (from `trainer.py` line 269-282):
```python
beta_clipped = np.clip(beta_np, sigma_min, sigma_max)  # [1e-3, 0.5]
beta_actual_min = beta_clipped.min()  # If all = 0.5, min = max = 0.5
beta_actual_max = beta_clipped.max()  # → beta_norm = 0.5 (uniform)
```

**Why Uniform Colors Appear:**
- If all σ values are clamped to 0.5 → `beta_actual_min == beta_actual_max`
- Normalization: `(0.5 - 0.5) / (0.5 - 0.5) = 0/0 → defaults to 0.5`
- Turbo colormap at 0.5 → **lime green/yellow** (middle of colormap)
- If all σ values are clamped to 1e-3 → appears as **blue** (low end of colormap)
- If all σ values are 0 (failed) → appears as **black**

---

## Root Cause: Why Sigma Inflates

Based on professor's notes and diagnostic output:

1. **Missing Learning Rate Reduction**: 
   - Professor's note: "Reduce σ learning rate (e.g., 0.1x the color LR) to avoid quick inflation"
   - **Check**: Is `uncertainty_lr_scale` set to 0.1 in your config?

2. **Initialization Too High**:
   - Professor's note: "Initialize log σ to s_0 = -3 (σ ≈ 0.05)"
   - If initialized higher (e.g., s_0 = -1 → σ ≈ 0.37), it starts closer to σ_max

3. **No Curriculum Learning**:
   - Professor's note: "Stage A: Train without uncertainty for first N_warm steps"
   - **Check**: Is `uncertainty_warmup_steps = 5000` working correctly?

4. **Weak Regularization**:
   - Professor's note: "Add regularizer R(σ) = (1/N) * Σ_r (log σ - log σ_0)²"
   - **Check**: Is `variance_regularizer` weight (β) set correctly?

---

## Recommended Fixes

### **Fix 1: Verify Learning Rate Scaling**

**Check your config** (`confs/replica_new.yaml` or similar):
```yaml
uncertainty_lr_scale: 0.1  # Should be 0.1x color LR (per professor's Stage B)
```

**If missing, add to optimizer setup** in `exp_runner.py` or `nd_uncertainty/trainer.py`:
```python
# Separate learning rate for uncertainty MLP (0.1x color LR)
uncertainty_params = list(self.uncertainty_pipeline.uncertainty_mlp.parameters())
color_params = [p for n, p in self.model.named_parameters() if 'color' in n.lower()]

# Get base LR for color
base_lr = self.optimizer.param_groups[0]['lr']
uncertainty_lr = base_lr * 0.1  # 0.1x color LR

# Add uncertainty params with reduced LR
self.optimizer.add_param_group({
    'params': uncertainty_params,
    'lr': uncertainty_lr
})
```

### **Fix 2: Improve Visualization to Show Clamp Status**

**Update `trainer.py` `_save_uncertainty_heatmap`** to detect and warn about clamping:

```python
def _save_uncertainty_heatmap(self, beta_image, acc_map, epoch, view_idx):
    beta_np = beta_image.numpy()
    acc_np = acc_map.numpy() if acc_map is not None else np.ones_like(beta_np)
    
    sigma_min = getattr(self.conf.loss, 'sigma_min', 1e-3)
    sigma_max = getattr(self.conf.loss, 'sigma_max', 0.5)
    
    # Detect clamping issues
    pct_at_min = (beta_np <= sigma_min + 1e-5).sum() / beta_np.size * 100
    pct_at_max = (beta_np >= sigma_max - 1e-5).sum() / beta_np.size * 100
    
    if pct_at_max > 50:
        print(f"[WARNING] {pct_at_max:.1f}% of sigma values at max clamp ({sigma_max}). "
              f"Sigma is inflating - check learning rate and regularization.")
    if pct_at_min > 50:
        print(f"[WARNING] {pct_at_min:.1f}% of sigma values at min clamp ({sigma_min}). "
              f"Sigma is collapsing - check initialization.")
    
    # Clamp to bounds
    beta_clipped = np.clip(beta_np, sigma_min, sigma_max)
    
    # Normalize using actual range (but warn if range is too small)
    beta_actual_min = beta_clipped.min()
    beta_actual_max = beta_clipped.max()
    beta_range = beta_actual_max - beta_actual_min
    
    if beta_range < 0.01:  # Very small range
        print(f"[WARNING] Sigma range is very small ({beta_range:.4f}). "
              f"Visualization will appear uniform. Values: min={beta_actual_min:.4f}, max={beta_actual_max:.4f}")
        # Use fixed bounds for visualization to show clamp status
        beta_norm = (beta_clipped - sigma_min) / (sigma_max - sigma_min + 1e-6)
    else:
        # Use actual range for better contrast
        beta_norm = (beta_clipped - beta_actual_min) / (beta_range + 1e-6)
    
    # Rest of visualization code...
```

### **Fix 3: Add Diagnostic Logging**

**Add to training loop** (in `exp_runner.py` or `nd_uncertainty/trainer.py`):

```python
# Log sigma statistics every N steps
if self.cur_step % 100 == 0 and 'beta' in sample:
    beta = sample['beta'].detach()
    sigma_min = getattr(self.conf.loss, 'sigma_min', 1e-3)
    sigma_max = getattr(self.conf.loss, 'sigma_max', 0.5)
    
    # Check clamp status
    pct_at_max = (beta >= sigma_max - 1e-5).float().mean().item() * 100
    pct_at_min = (beta <= sigma_min + 1e-5).float().mean().item() * 100
    
    self.loger.add_scalar('uncertainty/pct_at_max_clamp', pct_at_max, self.cur_step)
    self.loger.add_scalar('uncertainty/pct_at_min_clamp', pct_at_min, self.cur_step)
    self.loger.add_scalar('uncertainty/sigma_mean', beta.mean().item(), self.cur_step)
    self.loger.add_scalar('uncertainty/sigma_std', beta.std().item(), self.cur_step)
    
    if pct_at_max > 50:
        print(f"[WARNING] Step {self.cur_step}: {pct_at_max:.1f}% of sigma at max clamp. "
              f"Consider reducing uncertainty_lr_scale or increasing regularization.")
```

### **Fix 4: Verify Curriculum Learning**

**Check that warmup is working** (in `loss_wrapper.py` line 219):

```python
# Curriculum learning: warmup stage
if self.warmup_steps > 0 and cur_step is not None and cur_step < self.warmup_steps:
    # Disable uncertainty during warmup
    losses = self.base_loss(output, sample, prog)
    return losses
```

**Verify in config**:
```yaml
uncertainty_warmup_steps: 5000  # Should match professor's N_warm
```

---

## Experimental Checks

### **Check 1: Monitor Sigma Distribution**

**Create diagnostic script** (`check_sigma_distribution.py`):

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load checkpoint
checkpoint_path = "runs_new/replica_1/2025-12-12_17-45-47/checkpoints/epoch_2400.pth"
ckpt = torch.load(checkpoint_path, map_location='cpu')

# Extract uncertainty MLP state
if 'uncertainty_pipeline' in ckpt:
    mlp_state = ckpt['uncertainty_pipeline']['uncertainty_mlp']
    # Check if initialized correctly
    fc2_bias = mlp_state['fc2.bias'].item()
    print(f"Initial log_sigma (s_0): {fc2_bias:.4f}")
    print(f"Initial sigma (exp(s_0)): {np.exp(fc2_bias):.4f}")
    print(f"Expected: s_0 = -3.0, σ ≈ 0.05")
    
    if fc2_bias > -2.0:
        print("[WARNING] Initial log_sigma too high! Should be ~-3.0")
```

### **Check 2: Verify Learning Rate Scaling**

**Add to training script**:

```python
# After optimizer setup
for i, param_group in enumerate(self.optimizer.param_groups):
    if 'uncertainty' in str(param_group.get('params', [])):
        print(f"Uncertainty MLP LR: {param_group['lr']:.6f}")
    else:
        print(f"Base LR: {param_group['lr']:.6f}")

# Verify: uncertainty_lr should be ~0.1x base_lr
```

### **Check 3: Save Intermediate Uncertainty Maps**

**Modify `trainer.py` plot function** to save raw sigma values:

```python
# In plot() method, after render_beta_map:
beta_image, beta_stats = self.uncertainty_pipeline.render_beta_map(...)

# Save raw sigma values for analysis
np.save(
    os.path.join(self.plot_dir, 'uncertainty', f'epoch{epoch}_view{view_idx}_raw_sigma.npy'),
    beta_image.numpy()
)

# Log clamp statistics
sigma_min = getattr(self.conf.loss, 'sigma_min', 1e-3)
sigma_max = getattr(self.conf.loss, 'sigma_max', 0.5)
beta_np = beta_image.numpy()
pct_at_max = (beta_np >= sigma_max - 1e-5).sum() / beta_np.size * 100
print(f"Epoch {epoch}: {pct_at_max:.1f}% of sigma at max clamp ({sigma_max})")
```

### **Check 4: Compare Loss Components**

**Add to loss logging**:

```python
# In loss_wrapper.py forward():
if self.cur_step % 100 == 0:
    print(f"Step {self.cur_step}:")
    print(f"  Color loss (weighted): {L_color.item():.6f}")
    print(f"  Sigma mean: {sigma.mean().item():.6f}, std: {sigma.std().item():.6f}")
    print(f"  Sigma range: [{sigma.min().item():.6f}, {sigma.max().item():.6f}]")
    print(f"  Clamp bounds: [{self.sigma_min}, {self.sigma_max}]")
```

---

## Best Practices Summary

### **1. Clamping Strategy**
- ✅ **Clamp AFTER prediction, BEFORE loss**: Current implementation is correct
- ✅ **Use bounds [1e-3, 0.5]**: Matches professor's notes
- ⚠️ **Monitor clamp saturation**: If >50% values hit bounds, reduce LR or increase regularization

### **2. Learning Rate Management**
- ✅ **Reduce σ LR to 0.1x color LR**: Prevents quick inflation
- ✅ **Use curriculum learning**: Warmup 2000-5000 steps without uncertainty

### **3. Initialization**
- ✅ **Initialize s_0 = -3 (σ ≈ 0.05)**: Prevents underfitting
- ⚠️ **Don't initialize too high**: If s_0 > -2, σ starts too close to σ_max

### **4. Regularization**
- ✅ **Use variance regularizer**: R(σ) = (1/N) * Σ_r (log σ - log σ_0)²
- ✅ **Set β = 0.01**: Small weight to prevent trivial inflation

### **5. Visualization**
- ✅ **Normalize using actual min/max**: Better contrast
- ⚠️ **Detect and warn about uniform values**: Helps diagnose clamping issues
- ✅ **Use turbo colormap**: Standard for uncertainty visualization

---

## Quick Fix Checklist

- [ ] Verify `uncertainty_lr_scale: 0.1` in config
- [ ] Verify `uncertainty_warmup_steps: 5000` in config
- [ ] Verify `init_log_sigma: -3.0` in config
- [ ] Add clamp saturation logging to training loop
- [ ] Update visualization to detect uniform values
- [ ] Re-train scan 3 with fixed parameters
- [ ] Monitor sigma distribution during training (TensorBoard)

---

## Expected Behavior After Fixes

1. **Sigma Distribution**: Should spread across [1e-3, 0.5] range, not cluster at bounds
2. **Visualization**: Should show variation (not uniform colors)
3. **Training Stability**: Meshes should remain valid throughout training (not collapse after epoch 240)
4. **Loss Components**: Color loss should decrease as sigma learns proper uncertainty

---

## Next Steps

1. **Immediate**: Add diagnostic logging to identify why sigma inflates
2. **Short-term**: Verify and fix learning rate scaling for uncertainty MLP
3. **Medium-term**: Re-train scan 3 with corrected parameters
4. **Long-term**: Implement optional Stage C (conditional gating) for better uncertainty capacity
