"""
Uncertainty Trainer Integration

Subclasses ND-SDF's Trainer to inject β(r) computation before forward pass.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from exp_runner import Trainer as NDSDFTrainer
from nd_uncertainty.pipeline import UncertaintyPipeline


class UncertaintyTrainer(NDSDFTrainer):
    """
    Wraps ND-SDF's training loop so that before every forward pass,
    we compute β(r) using the new uncertainty pipeline.
    """

    def __init__(self, opt, gpu):
        super().__init__(opt, gpu)
        
        # Check if uncertainty is enabled
        use_uncertainty = getattr(self.conf.loss, 'use_uncertainty', True)
        
        if use_uncertainty:
            # Create uncertainty pipeline (DINO → patches → β(r))
            self.uncertainty_pipeline = UncertaintyPipeline(device=torch.device(f'cuda:{gpu}'))
            # DINO encoder is frozen by default in UncertaintyPipeline
            # Uncertainty MLP will be trainable (set to train mode when pipeline is in train mode)
            
            # Note: Uncertainty MLP will be lazily initialized on first forward pass
            # We'll add it to optimizer after first forward pass
            
            # Replace base loss with uncertainty-aware loss wrapper
            from nd_uncertainty.loss_wrapper import UncertaintyAwareLoss
            self.loss = UncertaintyAwareLoss(self.conf)
            # Set patch size for base loss (needed for depth loss, s3im, etc.)
            if hasattr(self.loss.base_loss, 'set_patch_size'):
                self.loss.base_loss.set_patch_size(self.conf.train.num_rays)
        else:
            # Uncertainty disabled - don't create pipeline
            self.uncertainty_pipeline = None

    def compute_uncertainty(self, sample):
        """
        Compute β(r) for the rays selected by ND-SDF's ray sampler.
        This must run BEFORE ND-SDF forward().
        
        Args:
            sample: dict with keys:
                - 'idx': (B,) image indices
                - 'sampling_idx': (B, R) ray indices into flattened image
                - 'h': (B,) image heights
                - 'w': (B,) image widths
        
        Returns:
            sample: updated with 'beta' and 'uncertainty_features'
        """
        # Skip if uncertainty is disabled
        if self.uncertainty_pipeline is None:
            # Set dummy values to avoid errors
            B, R = sample['sampling_idx'].shape
            device = sample['sampling_idx'].device
            sample['beta'] = torch.ones(B, R, 1, device=device)
            sample['uncertainty_features'] = torch.zeros(B, R, 384, device=device)  # Dummy features
            return sample
        # Get batch size and number of rays
        B = sample['idx'].shape[0]
        device = sample['idx'].device
        
        # Load full RGB images for each image in batch
        rgb_full_list = []
        heights = []
        widths = []
        
        for b in range(B):
            idx = sample['idx'][b].item()
            
            # Load full-resolution RGB image
            # Use the dataset's load_data method or load directly
            rgb = np.asarray(Image.open(self.train_dataset.rgb_paths[idx])).astype(np.float32) / 255.0
            
            # Get original image dimensions (before downscale)
            # Dataset stores downscaled dimensions, need to multiply by downscale factor
            original_h = int(self.train_dataset.h * self.train_downscale) if hasattr(self.train_dataset, 'h') else rgb.shape[0]
            original_w = int(self.train_dataset.w * self.train_downscale) if hasattr(self.train_dataset, 'w') else rgb.shape[1]
            
            # Resize if needed to match original resolution
            if rgb.shape[0] != original_h or rgb.shape[1] != original_w:
                import cv2
                rgb = cv2.resize(rgb, (original_w, original_h), interpolation=cv2.INTER_AREA)
            
            # Convert to tensor: (H, W, 3) -> (3, H, W) -> (1, 3, H, W)
            rgb_tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(device)
            rgb_full_list.append(rgb_tensor)
            heights.append(original_h)
            widths.append(original_w)
        
        # Stack to (B, 3, H, W)
        rgb_full = torch.cat(rgb_full_list, dim=0)  # (B, 3, H, W)
        
        # Get sampling indices and dimensions
        sampling_idx = sample['sampling_idx']  # (B, R)
        heights_tensor = torch.tensor(heights, device=device)  # (B,)
        widths_tensor = torch.tensor(widths, device=device)    # (B,)
        
        # Run the uncertainty pipeline
        # DINO encoder runs with no_grad, but MLP needs gradients
        with torch.no_grad():
            # DINO feature extraction (frozen)
            feature_maps = self.uncertainty_pipeline.dino_encoder(rgb_full)
            patches = self.uncertainty_pipeline.patch_sampler(
                feature_maps=feature_maps,
                sampling_idx=sampling_idx,
                heights=heights_tensor,
                widths=widths_tensor,
            )
        
        # Build MLP if needed
        self.uncertainty_pipeline._build_uncertainty_mlp_if_needed(patches)
        
        # MLP forward (needs gradients for training)
        beta = self.uncertainty_pipeline.uncertainty_mlp(patches)
        
        branch_out = {
            'beta': beta,
            'patch_features': patches,
        }
        
        # Attach results to sample dict
        sample['beta'] = branch_out['beta']  # (B, R, 1)
        sample['uncertainty_features'] = branch_out['patch_features']  # (B, R, C_patch)
        
        return sample

    def prepare_sample(self, sample, progress=0.0):
        """
        Override prepare_sample to inject β(r) computation.
        This is called BEFORE ND-SDF's forward pass.
        
        Note: ND-SDF's train_angle (delta-theta) computation happens in parent's
        prepare_sample() and update_train_angle(). Our uncertainty computation
        doesn't interfere with this - we only add beta to the sample dict.
        """
        # Call parent's prepare_sample first
        # This handles train_angle, sampling_idx, rays_o, rays_d, etc.
        sample = super().prepare_sample(sample, progress)
        
        # Compute uncertainty BEFORE forward pass (if enabled)
        if self.uncertainty_pipeline is not None:
            sample = self.compute_uncertainty(sample)
            
            # Add uncertainty MLP to optimizer if it was just initialized
            if (hasattr(self.uncertainty_pipeline, 'uncertainty_mlp') and 
                self.uncertainty_pipeline.uncertainty_mlp is not None and
                not any('uncertainty_mlp' in pg.get('name', '') for pg in self.optimizer.param_groups)):
                # Add uncertainty MLP parameters to optimizer
                self.optimizer.add_param_group({
                    'name': 'uncertainty_mlp',
                    'params': self.uncertainty_pipeline.uncertainty_mlp.parameters(),
                    'lr': self.conf.optim.lr,
                })
                if self.gpu == 0:
                    print("Added uncertainty MLP to optimizer")
        else:
            # Uncertainty disabled - set dummy values
            B, R = sample['sampling_idx'].shape
            device = sample['sampling_idx'].device
            sample['beta'] = torch.ones(B, R, 1, device=device)
            sample['uncertainty_features'] = torch.zeros(B, R, 384, device=device)
        
        return sample
    
    def train(self):
        """
        Override train to ensure uncertainty MLP is in train mode.
        """
        # Set uncertainty MLP to train mode (DINO encoder stays frozen)
        if hasattr(self.uncertainty_pipeline, 'uncertainty_mlp') and self.uncertainty_pipeline.uncertainty_mlp is not None:
            self.uncertainty_pipeline.uncertainty_mlp.train()
        
        # Verify delta-theta (train_angle) compatibility
        # ND-SDF's update_train_angle() is called in parent's train() method
        # Our uncertainty computation doesn't interfere with train_angle
        # train_angle is computed from output['angle'] which comes from ND-SDF's forward pass
        # We only add beta to sample, which doesn't affect angle computation
        
        # Call parent's train method (handles the main training loop)
        # Loss wrapper forwards set_curvature_weight and set_patch_size to base_loss
        super().train()
    
    def set_num_rays(self, max_num_rays, num_samples_per_ray, num_samples):
        """Override to forward set_patch_size to loss wrapper (which forwards to base_loss)."""
        super().set_num_rays(max_num_rays, num_samples_per_ray, num_samples)
        if hasattr(self.loss, 'set_patch_size'):
            self.loss.set_patch_size(self.num_rays)
