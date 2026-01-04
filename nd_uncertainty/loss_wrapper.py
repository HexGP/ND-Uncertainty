"""
Loss Wrapper for ND-SDF with Uncertainty

Wraps ND-SDF's ImplicitReconLoss to add uncertainty-weighted color loss
without modifying ND-SDF's core loss implementation.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from models.loss import ImplicitReconLoss
from nd_uncertainty.uncertainty_loss import (
    UncertaintyColorLoss,  # Legacy SSIM-based (kept for backward compatibility)
    heteroscedastic_color_loss,  # New heteroscedastic color loss
    uncertainty_regularizer,  # New uncertainty regularizer
)
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
            'variance_weight', 'use_uncertainty_annealing', 'uncertainty_anneal_param', 'weight_unc_sched',
            # New heteroscedastic uncertainty parameters
            'init_log_sigma', 'sigma_min', 'sigma_max', 'uncertainty_warmup_steps', 'uncertainty_lr_scale',
            # Hybrid approach parameter
            'hybrid_weight'
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
        
        # Initialize uncertainty components for heteroscedastic formulation
        # Get config values with defaults per ND-SDF hyperparameters
        # β = 0.01 (regularization weight, default per ND-SDF)
        self.unc_lambda_reg = getattr(conf.loss, 'unc_lambda_reg', 0.01)
        
        # s_0 = -3 → σ_0 ≈ 0.05 (initial log σ, default per ND-SDF)
        init_log_sigma = getattr(conf.loss, 'init_log_sigma', -3.0)
        self.sigma_0 = torch.exp(torch.tensor(init_log_sigma)).item()  # σ_0 = exp(s_0) ≈ 0.05
        
        # Clamping parameters: σ ∈ [1e-3, 0.5] (default per ND-SDF)
        self.sigma_min = getattr(conf.loss, 'sigma_min', 1e-3)
        self.sigma_max = getattr(conf.loss, 'sigma_max', 0.5)
        
        # Print sigma clamping parameters at initialization
        print(f"[UncertaintyAwareLoss] Initialized with sigma clamping:")
        print(f"  - sigma_min: {self.sigma_min}")
        print(f"  - sigma_max: {self.sigma_max}")
        print(f"  - unc_lambda_reg (β): {self.unc_lambda_reg}")
        print(f"  - sigma_0 (baseline): {self.sigma_0:.6f} (from init_log_sigma: {init_log_sigma})")
        
        # Legacy SSIM-based uncertainty loss (kept for backward compatibility)
        # The new heteroscedastic formulation is the default
        use_ssim = getattr(conf.loss, 'use_ssim_uncertainty', False)  # Default False - use heteroscedastic instead
        if use_ssim:
            # Only initialize if explicitly enabled (legacy mode)
            # For legacy mode, use old unc_clip_min parameter if available, otherwise use sigma_min
            uncer_clip_min = getattr(conf.loss, 'unc_clip_min', self.sigma_min)
            ssim_window_size = getattr(conf.loss, 'ssim_window_size', 5)
            stop_ssim_gradient = getattr(conf.loss, 'stop_ssim_gradient', True)
            ssim_anneal = getattr(conf.loss, 'ssim_anneal', 0.8)
            ssim_clip_max = getattr(conf.loss, 'ssim_clip_max', 5.0)
            self.unc_loss = UncertaintyColorLoss(
                lambda_reg=self.unc_lambda_reg,
                uncer_clip_min=uncer_clip_min,
                eps=getattr(conf.loss, 'unc_eps', 1e-3),
                use_ssim=use_ssim,
                ssim_window_size=ssim_window_size,
                stop_ssim_gradient=stop_ssim_gradient,
                ssim_anneal=ssim_anneal,
                ssim_clip_max=ssim_clip_max,
            )
        else:
            # New heteroscedastic formulation (default)
            self.unc_loss = None  # Not used in heteroscedastic mode
        
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
        
        # Weight for heteroscedastic color loss (λ_color in formula)
        # This replaces lambda_rgb_l1 when uncertainty is enabled (unless hybrid mode is used)
        # Default: λ_color = 1.0 (per ND-SDF hyperparameters)
        self.weight_unc = getattr(conf.loss, 'weight_unc', 1.0)
        
        # Hybrid approach: blend heteroscedastic + standard RGB L1
        # hybrid_weight = 0.0 → pure standard RGB L1 (no uncertainty)
        # hybrid_weight = 1.0 → pure heteroscedastic (replaces RGB L1, current approach)
        # hybrid_weight = 0.3 → 30% heteroscedastic, 70% standard RGB L1
        self.hybrid_weight = getattr(conf.loss, 'hybrid_weight', 1.0)  # Default 1.0 = pure heteroscedastic (backward compatible)
        
        # Store sigma clamping parameters for potential use
        self.sigma_min = getattr(conf.loss, 'sigma_min', 1e-3)
        self.sigma_max = getattr(conf.loss, 'sigma_max', 0.5)
        self.use_uncertainty_annealing = getattr(conf.loss, 'use_uncertainty_annealing', False)
        self.uncertainty_anneal_param = getattr(conf.loss, 'uncertainty_anneal_param', 0.8)
        
        # Curriculum learning: warmup stage (train without uncertainty for first N_warm steps)
        # This helps SDF, eikonal, and deflection field find good geometry before σ can "hide" errors
        # Default: N_warm = 5000 steps (per ND-SDF training schedule)
        self.warmup_steps = getattr(conf.loss, 'uncertainty_warmup_steps', 5000)
        
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
    
    def forward(self, output, sample, prog, cur_step=None):
        """
        Compute combined ND-SDF + heteroscedastic uncertainty loss.
        
        Per ND-SDF principles:
        - Heteroscedastic uncertainty ONLY for color loss
        - SDF, eikonal, normal losses remain deterministic (no uncertainty)
        - Total: L = λ_sdf*L_SDF + λ_eik*L_eik + λ_normal*L_normal + λ_color*Σ_r L_color(r) + β*R(σ)
        
        Curriculum learning: If warmup_steps > 0, uncertainty is disabled for first N_warm steps.
        This helps geometry find good initialization before σ can "hide" errors.
        
        Args:
            output: dict from ND-SDF forward pass
                - 'rgb': (B, R, 3) rendered RGB
                - 'outside': (B, R, 1) outside mask
            sample: dict containing:
                - 'rgb': (B, R, 3) ground truth RGB
                - 'beta': (B, R, 1) predicted uncertainty σ(r) (note: beta = sigma)
                - 'mask': (B, R, 1) optional foreground mask
            prog: training progress [0, 1]
            cur_step: current training step (for warmup check, optional)
        
        Returns:
            losses: dict with all ND-SDF losses + heteroscedastic color loss + regularizer + updated total
        """
        # Check if uncertainty is enabled
        if 'beta' not in sample:
            # Uncertainty disabled - use standard ND-SDF losses
            losses = self.base_loss(output, sample, prog)
            return losses
        
        # Curriculum learning: warmup stage
        # If we're in warmup period, disable uncertainty and use standard RGB loss
        if self.warmup_steps > 0 and cur_step is not None and cur_step < self.warmup_steps:
            losses = self.base_loss(output, sample, prog)
            return losses
        
        # Extract required tensors
        rgb_pred = output['rgb']          # (B, R, 3)
        rgb_gt = sample['rgb']            # (B, R, 3)
        sigma = sample['beta']             # (B, R, 1) - uncertainty σ (beta in code, sigma in formula)
        
        # CRITICAL: Check for NaN/Inf in sigma BEFORE clamping
        # If sigma has NaN values from MLP, they'll propagate and crash training
        if torch.isnan(sigma).any() or torch.isinf(sigma).any():
            print(f"[ERROR] NaN/Inf in sigma before clamp! Replacing with safe value.")
            sigma = torch.where(torch.isnan(sigma) | torch.isinf(sigma),
                               torch.tensor(self.sigma_max, device=sigma.device, dtype=sigma.dtype),
                               sigma)
        
        # CRITICAL: Clamp sigma_min for numerical stability (prevents division by zero in 1/σ²)
        # sigma_max is set very high (1000.0) - effectively no max clamp
        # Regularizer R(σ) = (log σ - log σ_0)² will prevent unbounded growth instead of hard clamping
        # This allows uncertainty to grow naturally for harder scenes while regularizer keeps it reasonable
        sigma = sigma.clamp(min=self.sigma_min, max=self.sigma_max)
        
        # Verify no NaN after clamping
        if torch.isnan(sigma).any():
            print(f"[ERROR] NaN still present after clamp! This should not happen.")
            sigma = torch.where(torch.isnan(sigma),
                               torch.tensor(self.sigma_max, device=sigma.device, dtype=sigma.dtype),
                               sigma)
        
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
        
        # --- ND-SDF Principle: Uncertainty ONLY for color, NOT for SDF/eikonal/normal ---
        # 
        # Step 1: Compute all deterministic losses (SDF, eikonal, normal, etc.)
        # These remain unchanged - no uncertainty weighting
        losses = self.base_loss(output, sample, prog)
        base_total = losses['total']
        
        # Step 2: Handle uncertainty loss based on mode (SSIM vs heteroscedastic)
        if self.unc_loss is not None:
            # SSIM-based uncertainty loss (baseline approach from replica_all8.yaml)
            # This is a SEPARATE loss that trains ONLY the uncertainty MLP
            # RGB L1 loss remains unchanged - uncertainty doesn't affect color loss
            # Formula: Luncer = LSSIM / (2β²) + λ1 log β
            # Gradients are stopped from SSIM to RGB to decouple training
            
            # Get training progress for SSIM annealing
            train_frac = prog if hasattr(prog, 'item') else float(prog)
            
            # Compute SSIM-based uncertainty loss
            L_unc, l_ssim_mean = self.unc_loss(rgb_pred, rgb_gt, sigma, mask=mask, train_frac=train_frac)
            
            # Add SSIM uncertainty loss as separate term (doesn't replace RGB L1)
            weight_unc = self.weight_unc_fn(prog)
            if self.use_uncertainty_annealing:
                uncer_rate = 1.0 + bias_function(prog, self.uncertainty_anneal_param)
                weight_unc = weight_unc * uncer_rate
            
            losses['uncertainty_loss'] = L_unc
            losses['ssim_uncertainty_loss'] = L_unc
            losses['l_ssim_mean'] = l_ssim_mean
            base_total = base_total + weight_unc * L_unc
            
            # SSIM mode doesn't use heteroscedastic regularizer (regularization is built into SSIM loss)
            # But we can optionally add variance regularizer if enabled
            if self.variance_reg is not None:
                # Variance regularizer uses DINO features, not sigma directly
                # This would need to be called from trainer if DINO features are available
                pass
        
        else:
            # Heteroscedastic color loss
            # Formula: L_color(r) = (1/(2σ²)) * ||C - Ĉ||² + (1/2) * log(σ²)
            L_heteroscedastic = heteroscedastic_color_loss(rgb_pred, rgb_gt, sigma, mask=mask)
            
            # Get the original RGB L1 loss weight
            lambda_rgb_l1 = self.base_loss.lambda_rgb_l1(prog)
            
            # Hybrid approach: blend heteroscedastic + standard RGB L1
            # hybrid_weight = 1.0 → pure heteroscedastic (replaces RGB L1, original approach)
            # hybrid_weight = 0.3 → 30% heteroscedastic, 70% standard RGB L1 (hybrid approach)
            # hybrid_weight = 0.0 → pure standard RGB L1 (no uncertainty)
            hybrid_weight = self.hybrid_weight
            
            if 'rgb_l1' in losses and lambda_rgb_l1 > 0:
                # Get standard RGB L1 loss
                L_rgb_l1 = losses['rgb_l1']
                
                # Hybrid blend: L_hybrid = α * L_heteroscedastic + (1-α) * L_RGB_L1
                # where α = hybrid_weight
                L_hybrid = hybrid_weight * L_heteroscedastic + (1.0 - hybrid_weight) * L_rgb_l1
                
                # Replace rgb_l1 contribution in total loss
                base_total = base_total - lambda_rgb_l1 * L_rgb_l1  # Remove original RGB L1
                base_total = base_total + lambda_rgb_l1 * L_hybrid  # Add hybrid loss
                
                # Store both losses for logging
                losses['rgb_l1'] = L_hybrid  # Store hybrid as rgb_l1 for logging
                losses['heteroscedastic_color_loss'] = L_heteroscedastic
                losses['rgb_l1_standard'] = L_rgb_l1  # Store original RGB L1 for reference
                losses['hybrid_color_loss'] = L_hybrid  # Store hybrid explicitly
            else:
                # If rgb_l1 was not used, just use heteroscedastic (fallback)
                weight_unc = self.weight_unc_fn(prog)
                if self.use_uncertainty_annealing:
                    uncer_rate = 1.0 + bias_function(prog, self.uncertainty_anneal_param)
                    weight_unc = weight_unc * uncer_rate
                losses['heteroscedastic_color_loss'] = L_heteroscedastic
                base_total = base_total + weight_unc * L_heteroscedastic
            
            # Step 3: Add uncertainty regularizer: R(σ) = (1/N) * Σ_r (log σ - log σ_0)²
            # Regularizer weight: β = unc_lambda_reg
            L_reg = uncertainty_regularizer(sigma, sigma_0=self.sigma_0, mask=mask)
            losses['uncertainty_regularizer'] = L_reg
            base_total = base_total + self.unc_lambda_reg * L_reg
            
            # For backward compatibility, also store as uncertainty_loss
            losses['uncertainty_loss'] = L_heteroscedastic  # Heteroscedastic color loss
        
        # Step 4: Update total loss
        losses['total'] = base_total
        
        return losses
