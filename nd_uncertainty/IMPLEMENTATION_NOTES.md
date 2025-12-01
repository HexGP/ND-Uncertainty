# Implementation Notes: Uncertainty Loss Integration

## Comparison with NeRF-on-the-Go and ChatGPT's Recommendation

### What ChatGPT Recommended:
- Simple uncertainty loss: `L = ||C - Ĉ||² / (2β²) + λ log β`
- Basic loss wrapper that adds uncertainty term to ND-SDF's total loss
- No SSIM integration (mentioned as optional future step)

### What NeRF-on-the-Go Actually Does:
Looking at `nerf-on-the-go/internal/train_utils.py` lines 159-179:

1. **Clips uncertainty**: `uncer = jnp.clip(uncer, a_min=config.uncer_clip_min)+1e-3`
   - Default `uncer_clip_min = 0.1` (line 183 in configs.py)
   - Adds `1e-3` epsilon for stability

2. **Uses SSIM in addition to RGB MSE**:
   - Computes SSIM components: `l, c, s = compute_ssim(...)`
   - SSIM loss: `my_ssim_loss = rate * (1-l)*(1-s)*(1-c)`
   - SSIM uncertainty loss: `ssim_loss = my_ssim_loss / uncer² + reg_mult * log(uncer)`
   - RGB MSE loss: `data_loss = 0.5 * resid_sq / uncer²`
   - Combined: `data_loss += config.ssim_mult * ssim_loss`

3. **Training progress-based adjustments**:
   - Adjusts uncertainty rate based on `train_frac`
   - Uses bias function for annealing

4. **Regularization weight**: `reg_mult = 0.5` (default, line 181 in configs.py)

### What I Implemented:

**File: `nd_uncertainty/uncertainty_loss.py`**

✅ **Matches NeRF-on-the-Go:**
- Clips uncertainty: `beta.clamp(min=uncer_clip_min) + eps` (matches line 161)
- Uses `0.5 * residual_sq / beta²` for RGB term (matches line 178)
- Uses `lambda_reg * log(beta)` for regularization (matches line 173)
- Default `lambda_reg=0.5` (matches `reg_mult=0.5` in NeRF-OTG)
- Default `uncer_clip_min=0.1` (matches NeRF-OTG config)
- Default `eps=1e-3` (matches NeRF-OTG)

❌ **Differs from NeRF-on-the-Go:**
- **No SSIM integration** (ChatGPT's recommendation was to keep it simple for MVP)
- **No training progress-based uncertainty adjustment** (simpler for initial integration)
- **No separate SSIM loss term** (only RGB MSE weighted by uncertainty)

**File: `nd_uncertainty/loss_wrapper.py`**

✅ **Matches ChatGPT's recommendation:**
- Wraps `ImplicitReconLoss` without modifying ND-SDF core
- Adds uncertainty loss to total: `losses['total'] += weight_unc * L_unc`
- Forwards methods like `set_patch_size` and `set_curvature_weight` to base_loss

✅ **Additional features beyond ChatGPT:**
- Supports sequential learning rate scheduling for `weight_unc`
- Proper mask handling (foreground + not outside)
- Configurable via OmegaConf (matches ND-SDF's config style)

**File: `nd_uncertainty/trainer.py`**

✅ **Matches ChatGPT's recommendation:**
- Replaces `self.loss` with `UncertaintyAwareLoss` in `__init__`
- Forwards loss wrapper methods to base_loss

### Summary of Differences:

1. **Simpler than NeRF-on-the-Go**: 
   - No SSIM (can be added later as optional extension)
   - No training progress-based uncertainty adjustment
   - Focus on core uncertainty-weighted RGB loss

2. **More aligned with NeRF in the Wild paper**:
   - Core formula: `L = ||C - Ĉ||² / (2β²) + λ log β`
   - This is the fundamental uncertainty loss from NeRF in the Wild

3. **Matches NeRF-on-the-Go's numerical stability**:
   - Same clipping strategy
   - Same epsilon values
   - Same regularization weight defaults

### Why This Approach:

- **MVP first**: Start with core uncertainty loss (matches professor's "minimum viable integration")
- **Easy to extend**: SSIM can be added later as an optional component
- **Matches NeRF in the Wild**: The core formula is correct
- **Numerically stable**: Uses NeRF-on-the-Go's proven clipping/epsilon strategy

### Future Extensions (Optional):

1. **SSIM-based uncertainty loss** (from NeRF-on-the-Go lines 162-173):
   ```python
   # Add SSIM computation
   l, c, s = compute_ssim(rgb_pred, rgb_gt)
   ssim_loss = rate * (1-l)*(1-s)*(1-c) / beta² + lambda_reg * log(beta)
   ```

2. **Patch variance regularization** (from NeRF-on-the-Go lines 81-85):
   ```python
   # DINO patch similarity variance (eq 2 & 3 in paper)
   dino_var_loss = mean(uncer_variances)
   ```

3. **Training progress-based adjustments** (from NeRF-on-the-Go lines 176-177):
   ```python
   # Adjust uncertainty rate based on training progress
   uncer_rate = 1 + bias(train_frac, ssim_anneal)
   ```
