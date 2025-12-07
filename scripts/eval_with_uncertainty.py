"""
Quick evaluation script to compute metrics (PSNR, SSIM, LPIPS) with uncertainty enabled.
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.system import ImplicitReconSystem
from dataset.base_dataset import BaseDataset
from models.loss import get_psnr
from models.metrics.ssim import SSIM
import utils.utils as utils
from nd_uncertainty.pipeline import UncertaintyPipeline

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("[Warning] LPIPS not available. Install with: pip install lpips")


def compute_metrics(pred_rgb, gt_rgb, mask=None):
    """
    Compute PSNR, SSIM, and LPIPS metrics.
    
    Args:
        pred_rgb: (H, W, 3) or (B, H, W, 3) predicted RGB
        gt_rgb: (H, W, 3) or (B, H, W, 3) ground truth RGB
        mask: (H, W) or (B, H, W) optional mask
    
    Returns:
        dict with psnr, ssim, lpips
    """
    # Convert to torch tensors if needed
    if isinstance(pred_rgb, np.ndarray):
        pred_rgb = torch.from_numpy(pred_rgb).float()
    if isinstance(gt_rgb, np.ndarray):
        gt_rgb = torch.from_numpy(gt_rgb).float()
    
    # Ensure same shape
    if pred_rgb.dim() == 3:  # (H, W, 3)
        pred_rgb = pred_rgb.unsqueeze(0)  # (1, H, W, 3)
        gt_rgb = gt_rgb.unsqueeze(0)  # (1, H, W, 3)
    
    # Convert to (B, C, H, W) for SSIM and LPIPS
    pred_tensor = pred_rgb.permute(0, 3, 1, 2).clamp(0, 1)  # (B, 3, H, W)
    gt_tensor = gt_rgb.permute(0, 3, 1, 2).clamp(0, 1)  # (B, 3, H, W)
    
    # Apply mask if provided
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
        if mask.dim() == 2:  # (H, W)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        elif mask.dim() == 3:  # (B, H, W)
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
        pred_tensor = pred_tensor * mask
        gt_tensor = gt_tensor * mask
    
    # PSNR
    # Flatten for PSNR computation
    pred_flat = pred_rgb.view(-1, 3)  # (B*H*W, 3)
    gt_flat = gt_rgb.view(-1, 3)  # (B*H*W, 3)
    if mask is not None:
        mask_flat = mask.view(-1)  # (B*H*W,)
        if mask_flat.sum() > 0:
            pred_flat = pred_flat[mask_flat > 0]
            gt_flat = gt_flat[mask_flat > 0]
    
    psnr = get_psnr(pred_flat, gt_flat, mask=None)
    
    # SSIM
    ssim_fn = SSIM(window_size=11, size_average=True)
    if pred_tensor.is_cuda:
        ssim_fn = ssim_fn.cuda()
    ssim_val = ssim_fn(pred_tensor, gt_tensor).item()
    
    # LPIPS
    lpips_val = None
    if LPIPS_AVAILABLE:
        lpips_fn = lpips.LPIPS(net='alex').eval()
        if pred_tensor.is_cuda:
            lpips_fn = lpips_fn.cuda()
        with torch.no_grad():
            lpips_val = lpips_fn(pred_tensor, gt_tensor).item()
    
    return {
        'psnr': psnr.item() if isinstance(psnr, torch.Tensor) else psnr,
        'ssim': ssim_val,
        'lpips': lpips_val
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with uncertainty')
    parser.add_argument('--conf', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--scan_id', type=str, default=None, help='Scan ID (overrides config)')
    parser.add_argument('--data_dir', type=str, default='', help='Data directory (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for rendered images')
    parser.add_argument('--downscale', type=int, default=1, help='Downscale factor')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    args = parser.parse_args()
    
    # Load config
    conf = OmegaConf.load(args.conf)
    if args.scan_id:
        conf.dataset.scan_id = args.scan_id
    if args.data_dir:
        conf.dataset.data_dir = args.data_dir
    
    device = f'cuda:{args.gpu}'
    bound = 1.0 if not hasattr(conf.model, 'bound') else conf.model.bound
    
    # Load model (uncertainty doesn't affect model loading, just loss computation)
    print(f"Loading model from {args.checkpoint} on {device}...")
    # Load checkpoint first to avoid device mismatch
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = ImplicitReconSystem(conf, bound, device=device)
    model = model.to(device)
    cur_step = ckpt.get('step', 1e9)
    
    # Restore model state
    model.load_state_dict(ckpt['model'])
    if conf.model.object.sdf.enable_progressive:
        model.sdf.set_active_levels(cur_step)
        model.sdf.set_normal_epsilon()
    if conf.model.background.enabled and conf.model.background.type == 'grid_nerf':
        model.bg_nerf.set_active_levels(cur_step)
    model.eval()
    
    use_uncertainty = getattr(conf.loss, 'use_uncertainty', False)
    
    # Initialize uncertainty pipeline if enabled
    uncertainty_pipeline = None
    if use_uncertainty:
        print("[Info] Uncertainty is enabled - computing beta statistics")
        if hasattr(conf, 'uncertainty'):
            patch_size = getattr(conf.uncertainty, 'patch_size', 7)
            dilation = getattr(conf.uncertainty, 'dilation', 2)
            max_chunk_rays = getattr(conf.uncertainty, 'max_chunk_rays', 16384)
        else:
            patch_size = 7
            dilation = 2
            max_chunk_rays = 16384
        
        uncertainty_pipeline = UncertaintyPipeline(
            patch_size=patch_size,
            dilation=dilation,
            device=device
        )
        uncertainty_pipeline.eval()
        
        # Note: Uncertainty MLP will be initialized lazily on first forward pass
        # The trained weights should be in the model state dict if saved properly
        # For now, we'll let it initialize fresh (will use trained weights if in model state)
    
    # Load validation dataset
    print(f"Loading validation dataset for scan {conf.dataset.scan_id}...")
    valid_dataset = BaseDataset(
        conf.dataset,
        split='valid',
        num_rays=conf.train.num_rays,
        downscale=args.downscale,
        fewshot=getattr(conf.dataset, 'fewshots', False),
        fewshot_idx=getattr(conf.dataset, 'fewshot_idx', [])
    )
    valid_dataset.set_loop_all()
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )
    
    # Compute metrics
    print(f"Evaluating on {len(valid_dataset)} validation images...")
    all_metrics = []
    
    for i, sample in enumerate(tqdm(valid_loader, desc="Evaluating")):
        sample = {k: v.to(device) for k, v in sample.items()}
        split_sample = utils.split_input(sample, valid_dataset.total_pixels, n_pixels=1024)
        
        outputs = []
        for s in split_sample:
            with torch.no_grad():
                output = model(s)
            outputs.append({
                'rgb': output['rgb'].detach(),
                'outside': output.get('outside', None),
            })
        
        # Merge outputs
        merged_outputs = utils.merge_output(outputs)
        pred_rgb = merged_outputs['rgb'][0].cpu().numpy()  # (H*W, 3)
        gt_rgb = sample['rgb'][0].cpu().numpy()  # (H*W, 3)
        
        # Reshape to (H, W, 3)
        H, W = valid_dataset.h, valid_dataset.w
        pred_rgb = pred_rgb.reshape(H, W, 3)
        gt_rgb = gt_rgb.reshape(H, W, 3)
        
        # Get mask if available
        mask = None
        if 'mask' in sample:
            mask = sample['mask'][0].cpu().numpy().reshape(H, W)
        elif 'outside' in merged_outputs:
            outside = merged_outputs['outside'][0].cpu().numpy().reshape(H, W)
            mask = ~outside  # Foreground mask
        
        # Compute RGB metrics
        metrics = compute_metrics(pred_rgb, gt_rgb, mask=mask)
        
        # Compute uncertainty metrics if enabled
        if uncertainty_pipeline is not None:
            # Load full RGB image for DINO
            idx = sample['idx'][0].item()
            rgb_full = np.asarray(Image.open(valid_dataset.rgb_paths[idx]))
            rgb_full = torch.from_numpy(rgb_full).float().to(device) / 255.0  # (H, W, 3)
            rgb_full = rgb_full.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            
            # Extract DINO features
            with torch.no_grad():
                feature_maps = uncertainty_pipeline.dino_encoder(rgb_full)
            
            # Render beta map
            beta_image, beta_stats = uncertainty_pipeline.render_beta_map(
                feature_maps=feature_maps,
                H=H,
                W=W,
                device=device,
                max_chunk_rays=max_chunk_rays if hasattr(conf, 'uncertainty') else 16384
            )
            
            # Add uncertainty metrics
            metrics['beta_mean'] = beta_stats['mean']
            metrics['beta_median'] = beta_stats['median']
            metrics['beta_std'] = beta_stats['std']
            metrics['beta_min'] = beta_stats['min']
            metrics['beta_max'] = beta_stats['max']
            metrics['beta_p25'] = beta_stats['p25']
            metrics['beta_p50'] = beta_stats['p50']
            metrics['beta_p75'] = beta_stats['p75']
            metrics['beta_p95'] = beta_stats['p95']
        
        all_metrics.append(metrics)
        
        # Save rendered image if output_dir specified
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            idx = sample['idx'][0].item()
            filename = os.path.basename(valid_dataset.rgb_paths[idx])
            pred_img = Image.fromarray((pred_rgb * 255).astype(np.uint8))
            pred_img.save(os.path.join(args.output_dir, f'pred_{filename}'))
    
    # Aggregate metrics
    avg_metrics = {
        'psnr': np.mean([m['psnr'] for m in all_metrics]),
        'ssim': np.mean([m['ssim'] for m in all_metrics]),
    }
    if all(m['lpips'] is not None for m in all_metrics):
        avg_metrics['lpips'] = np.mean([m['lpips'] for m in all_metrics if m['lpips'] is not None])
    
    # Aggregate uncertainty metrics if available
    if use_uncertainty and 'beta_mean' in all_metrics[0]:
        avg_metrics['beta_mean'] = np.mean([m['beta_mean'] for m in all_metrics])
        avg_metrics['beta_median'] = np.mean([m['beta_median'] for m in all_metrics])
        avg_metrics['beta_std'] = np.mean([m['beta_std'] for m in all_metrics])
        avg_metrics['beta_min'] = np.min([m['beta_min'] for m in all_metrics])
        avg_metrics['beta_max'] = np.max([m['beta_max'] for m in all_metrics])
        avg_metrics['beta_p25'] = np.mean([m['beta_p25'] for m in all_metrics])
        avg_metrics['beta_p50'] = np.mean([m['beta_p50'] for m in all_metrics])
        avg_metrics['beta_p75'] = np.mean([m['beta_p75'] for m in all_metrics])
        avg_metrics['beta_p95'] = np.mean([m['beta_p95'] for m in all_metrics])
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Number of images: {len(all_metrics)}")
    print(f"\nRGB Metrics:")
    print(f"  PSNR:  {avg_metrics['psnr']:.4f} dB")
    print(f"  SSIM:  {avg_metrics['ssim']:.4f}")
    if 'lpips' in avg_metrics:
        print(f"  LPIPS: {avg_metrics['lpips']:.4f}")
    
    if use_uncertainty and 'beta_mean' in avg_metrics:
        print(f"\nUncertainty Metrics (β values):")
        print(f"  Mean:   {avg_metrics['beta_mean']:.4f}")
        print(f"  Median: {avg_metrics['beta_median']:.4f}")
        print(f"  Std:    {avg_metrics['beta_std']:.4f}")
        print(f"  Min:    {avg_metrics['beta_min']:.4f}")
        print(f"  Max:    {avg_metrics['beta_max']:.4f}")
        print(f"  P25:    {avg_metrics['beta_p25']:.4f}")
        print(f"  P50:    {avg_metrics['beta_p50']:.4f}")
        print(f"  P75:    {avg_metrics['beta_p75']:.4f}")
        print(f"  P95:    {avg_metrics['beta_p95']:.4f}")
    print("="*50)
    
    # Print per-image metrics (abbreviated if many images)
    if len(all_metrics) <= 20:
        print("\nPer-image metrics:")
        for i, m in enumerate(all_metrics):
            idx = valid_dataset.rgb_paths[i].split('/')[-1] if i < len(valid_dataset.rgb_paths) else f"img_{i}"
            line = f"  {idx}: PSNR={m['psnr']:.4f}, SSIM={m['ssim']:.4f}"
            if m['lpips'] is not None:
                line += f", LPIPS={m['lpips']:.4f}"
            if use_uncertainty and 'beta_mean' in m:
                line += f", β_mean={m['beta_mean']:.4f}"
            print(line)
    else:
        print(f"\n(Per-image metrics omitted for {len(all_metrics)} images)")


if __name__ == '__main__':
    main()

