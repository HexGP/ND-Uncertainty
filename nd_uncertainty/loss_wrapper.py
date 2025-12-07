"""
Loss Wrapper for ND-SDF with Uncertainty

Wraps ND-SDF's ImplicitReconLoss to add uncertainty-weighted color loss
without modifying ND-SDF's core loss implementation.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.loss import ImplicitReconLoss
from nd_uncertainty.uncertainty_loss import UncertaintyColorLoss
from nd_uncertainty.variance_regularizer import PatchVarianceRegularizer
from nd_uncertainty.ssim_utils import bias_function


class UncertaintyAwareLoss(nn.Module):
    """
    Wraps ND-SDF's loss aggregator and adds:
        + L_uncertainty(C, Ĉ, β)
        + (optional) patch variance regularization (future)
    
    This allows us to use the original ND-SDF model unchanged.
    """
    
    def __init__(self, conf):
        """
        Args:
            conf: OmegaConf configuration object (ND-SDF's config)
        """
        super().__init__()
        
        # Initialize base ND-SDF loss with all its native components
        # Filter out uncertainty-related parameters that ImplicitReconLoss doesn't know about
        loss_conf = dict(conf.loss)
        uncertainty_params = [
            'use_uncertainty', 'use_ssim_uncertainty', 'use_variance_regularizer',
            'weight_unc', 'unc_lambda_reg', 'unc_clip_min', 'unc_eps',
            'ssim_window_size', 'stop_ssim_gradient', 'ssim_anneal', 'ssim_clip_max',
            'variance_weight', 'use_uncertainty_annealing', 'uncertainty_anneal_param', 'weight_unc_sched'
        ]
        for param in uncertainty_params:
            loss_conf.pop(param, None)  # Remove if exists, ignore if doesn't
        
        self.base_loss = ImplicitReconLoss(**loss_conf, optim_conf=conf.optim)
        
        # Check if uncertainty is enabled
        use_uncertainty = getattr(conf.loss, 'use_uncertainty', True)
        
        if not use_uncertainty:
            # Uncertainty disabled - just use base loss
            self.unc_loss = None
            self.variance_reg = None
            self.weight_unc_fn = lambda prog: 0.0
            self.use_uncertainty_annealing = False
            return
        
        # Initialize uncertainty loss (ND-SDF paper approach)
        # Get config values with defaults matching ND-SDF paper
        lambda_reg = getattr(conf.loss, 'unc_lambda_reg', 0.5)  # λ1 in paper Eq. 9
        uncer_clip_min = getattr(conf.loss, 'unc_clip_min', 0.1)
        use_ssim = getattr(conf.loss, 'use_ssim_uncertainty', True)  # Default True per paper
        ssim_window_size = getattr(conf.loss, 'ssim_window_size', 5)
        stop_ssim_gradient = getattr(conf.loss, 'stop_ssim_gradient', True)  # Default True for decoupling
        ssim_anneal = getattr(conf.loss, 'ssim_anneal', 0.8)  # SSIM rate annealing (matches NeRF-on-the-Go)
        ssim_clip_max = getattr(conf.loss, 'ssim_clip_max', 5.0)  # SSIM loss clipping (matches NeRF-on-the-Go)
        
        self.unc_loss = UncertaintyColorLoss(
            lambda_reg=lambda_reg,
            uncer_clip_min=uncer_clip_min,
            eps=getattr(conf.loss, 'unc_eps', 1e-3),
            use_ssim=use_ssim,
            ssim_window_size=ssim_window_size,
            stop_ssim_gradient=stop_ssim_gradient,
            ssim_anneal=ssim_anneal,
            ssim_clip_max=ssim_clip_max,
        )
        
        # Initialize patch variance regularizer (ND-SDF paper Eq. 2 & 3)
        # This regularizer is fine to use - it doesn't break decoupling because:
        # - It only uses β values and DINO features (not ND-SDF predictions)
        # - It doesn't cause gradients to flow from uncertainty loss to NeRF model
        # - It's a purely internal smoothing mechanism for uncertainty MLP
        use_variance_reg = getattr(conf.loss, 'use_variance_regularizer', True)  # Default True per paper
        variance_weight = getattr(conf.loss, 'variance_weight', 0.1)
        if use_variance_reg:
            self.variance_reg = PatchVarianceRegularizer(
                top_k=128,
                similarity_threshold=0.75,
                weight=variance_weight,
            )
        else:
            self.variance_reg = None
        
        # Weight for uncertainty loss (can be scheduled with training progress)
        self.weight_unc = getattr(conf.loss, 'weight_unc', 1.0)
        self.use_uncertainty_annealing = getattr(conf.loss, 'use_uncertainty_annealing', False)
        self.uncertainty_anneal_param = getattr(conf.loss, 'uncertainty_anneal_param', 0.8)
        
        # Optional: sequential learning rate for uncertainty loss
        if hasattr(conf.loss, 'weight_unc_sched'):
            # Support sequential LR scheduling like ND-SDF's other losses
            from functools import partial
            def sequential_lr(progress, lr):
                if isinstance(lr, (int, float)):
                    return lr
                return lr[2] + min(1.0, max(0.0, (progress - lr[0]) / (lr[1] - lr[0]))) * (lr[3] - lr[2])
            self.weight_unc_fn = partial(sequential_lr, lr=conf.loss.weight_unc_sched)
        else:
            self.weight_unc_fn = lambda prog: self.weight_unc
    
    def set_patch_size(self, num_rays):
        """Forward set_patch_size to base_loss (called by trainer)."""
        if hasattr(self.base_loss, 'set_patch_size'):
            self.base_loss.set_patch_size(num_rays)
    
    def set_curvature_weight(self, cur_step, anneal_levels, grow_rate):
        """Forward set_curvature_weight to base_loss (called by trainer)."""
        if hasattr(self.base_loss, 'set_curvature_weight'):
            self.base_loss.set_curvature_weight(cur_step, anneal_levels, grow_rate)
    
    def __getattr__(self, name):
        """
        Forward attribute access to base_loss for attributes that don't exist
        on UncertaintyAwareLoss (e.g., lambda_curvature, set_curvature_weight, etc.).

        Pattern:
          1) Let PyTorch / nn.Module resolve attributes first (parameters, buffers,
             registered submodules, including base_loss).
          2) If that fails, forward the lookup to base_loss via self._modules.
        """
        # 1. Try PyTorch's built-in lookup first. This checks _parameters,
        # _buffers and _modules, so registered submodules like base_loss are
        # resolved here if possible.
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass

        # 2. If PyTorch couldn't resolve it, safely forward to base_loss.
        # Access via _modules to avoid recursion and ensure the module is registered.
        if "base_loss" in self._modules:
            base = self._modules["base_loss"]
            if hasattr(base, name):
                return getattr(base, name)

        # 3. If neither the wrapper nor base_loss have it, raise a normal error.
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def forward(self, output, sample, prog):
        """
        Compute combined ND-SDF + uncertainty loss.
        
        Args:
            output: dict from ND-SDF forward pass
                - 'rgb': (B, R, 3) rendered RGB
                - 'outside': (B, R, 1) outside mask
            sample: dict containing:
                - 'rgb': (B, R, 3) ground truth RGB
                - 'beta': (B, R, 1) predicted uncertainty β(r)
                - 'mask': (B, R, 1) optional foreground mask
            prog: training progress [0, 1]
        
        Returns:
            losses: dict with all ND-SDF losses + uncertainty_loss + updated total
        """
        # Run all ND-SDF native losses (eikonal, RGB, depth, normal, etc.)
        losses = self.base_loss(output, sample, prog)
        
        # Check if uncertainty is enabled
        if self.unc_loss is None or 'beta' not in sample:
            # Uncertainty disabled - return base losses only
            return losses
        
        # --- Uncertainty loss ---
        # Extract required tensors
        rgb_pred = output['rgb']          # (B, R, 3)
        rgb_gt = sample['rgb']             # (B, R, 3)
        beta = sample['beta']              # (B, R, 1)
        
        # Get mask (foreground + not outside)
        outside = output.get('outside', None)  # (B, R, 1)
        foreground_mask = sample.get('mask', None)  # (B, R, 1)
        
        # Combine masks: foreground AND not outside
        if outside is not None and foreground_mask is not None:
            mask = (~outside) & foreground_mask  # (B, R, 1)
        elif outside is not None:
            mask = ~outside  # (B, R, 1)
        elif foreground_mask is not None:
            mask = foreground_mask  # (B, R, 1)
        else:
            mask = None
        
        # Compute SSIM-based uncertainty loss (ND-SDF paper Eq. 9)
        # This loss trains ONLY the uncertainty MLP, not the NeRF model
        # Gradients are stopped from flowing back to RGB predictions (decoupling)
        # Pass train_frac (prog) for SSIM rate scaling (100-1000x multiplier)
        L_unc, L_ssim_mean = self.unc_loss(rgb_pred, rgb_gt, beta, mask=mask, train_frac=prog)
        
        # Compute patch variance regularization (ND-SDF paper Eq. 2 & 3)
        # This regularizer encourages spatial-temporal consistency in uncertainty predictions
        # IMPORTANT: L_reg is added to L_unc (not to total loss separately)
        # This ensures it only influences Uncertainty MLP branch, not ND-SDF training
        L_var = None
        if self.variance_reg is not None and 'uncertainty_features' in sample:
            patch_features = sample['uncertainty_features']  # (B, R, C_patch)
            L_var = self.variance_reg(patch_features, beta)
            losses['variance_regularizer'] = L_var
            # Add L_reg to uncertainty loss (not to total loss separately)
            # This way L_reg only affects Uncertainty MLP, not ND-SDF
            L_unc = L_unc + L_var
        
        # Get uncertainty loss weight (may be scheduled)
        weight_unc = self.weight_unc_fn(prog)
        
        # Apply training progress-based uncertainty annealing (optional)
        if self.use_uncertainty_annealing:
            # Adjust uncertainty weight based on training progress
            uncer_rate = 1.0 + bias_function(prog, self.uncertainty_anneal_param)
            weight_unc = weight_unc * uncer_rate
        
        # ND-SDF Paper Approach: DECOUPLED Training
        # 
        # Key principle: Uncertainty MLP training is SEPARATE from NeRF training
        # - ND-SDF trains with standard RGB loss (kept above, not replaced)
        # - Uncertainty MLP trains with SSIM-based uncertainty loss + L_reg (added here)
        # - Gradients are stopped to prevent coupling (handled in UncertaintyColorLoss)
        #
        # This ensures:
        # 1. Uncertainty prediction doesn't influence NeRF color rendering
        # 2. NeRF color errors don't directly affect uncertainty MLP training
        # 3. Uncertainty learns to detect structural differences (distractors) via SSIM
        # 4. L_reg only affects Uncertainty MLP (beta comes from MLP, DINO is frozen)
        
        # Add uncertainty loss (which now includes L_reg) to total
        # This only affects uncertainty MLP gradients (RGB is detached in UncertaintyColorLoss)
        losses['uncertainty_loss'] = L_unc
        losses['l_ssim'] = L_ssim_mean  # LSSIM value for logging (Eq. 8)
        # Add uncertainty loss to existing ND-SDF total (not replacing, just adding)
        base_total = losses['total']  # This is the sum of all ND-SDF losses (rgb_l1, rgb_mse, eik, smooth, curvature, etc.)
        losses['total'] = base_total + weight_unc * L_unc
        
        return losses
