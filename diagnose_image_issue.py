#!/usr/bin/env python3
"""
Diagnose image generation issues - check if images are blank/black
"""

import os
import numpy as np
from PIL import Image
import glob

def analyze_image(image_path):
    """Analyze an image to see if it's blank/black"""
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        # Check if grayscale or RGB
        if len(img_array.shape) == 2:
            # Grayscale
            mean_val = img_array.mean()
            std_val = img_array.std()
            min_val = img_array.min()
            max_val = img_array.max()
            unique_vals = len(np.unique(img_array))
        else:
            # RGB or RGBA
            mean_val = img_array.mean()
            std_val = img_array.std()
            min_val = img_array.min()
            max_val = img_array.max()
            unique_vals = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
        
        # Determine status
        if mean_val < 5 and std_val < 5:
            status = "BLANK/BLACK"
        elif unique_vals < 10:
            status = "NEARLY UNIFORM"
        elif std_val < 20:
            status = "LOW VARIANCE"
        else:
            status = "NORMAL"
        
        return {
            'path': image_path,
            'size': img.size,
            'mode': img.mode,
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'unique_colors': unique_vals,
            'status': status
        }
    except Exception as e:
        return {'path': image_path, 'error': str(e)}

def compare_replicas():
    """Compare images between replica_1 and replica_2"""
    base_dir = "runs_new"
    
    replicas = {
        'replica_1': 'replica_1/2025-12-12_17-45-47/plots/rgb',
        'replica_2': 'replica_2/2025-12-12_17-46-44/plots/rgb',
        'replica_3': 'replica_3/2025-12-12_17-47-18/plots/rgb',
        'replica_5': 'replica_5/2025-12-13_16-24-05/plots/rgb',
        'replica_6': 'replica_6/2025-12-13_16-24-40/plots/rgb',
    }
    
    print("="*80)
    print("IMAGE GENERATION DIAGNOSTIC")
    print("="*80)
    
    for replica_name, rel_path in replicas.items():
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            print(f"\n{replica_name}: Directory not found")
            continue
        
        print(f"\n{replica_name}:")
        print("-" * 80)
        
        # Get all PNG files
        image_files = glob.glob(os.path.join(full_path, "*.png"))
        if not image_files:
            print("  No images found")
            continue
        
        # Analyze first 3 images
        image_files.sort()
        for img_file in image_files[:3]:
            result = analyze_image(img_file)
            if 'error' in result:
                print(f"  {os.path.basename(img_file)}: ERROR - {result['error']}")
            else:
                print(f"  {os.path.basename(img_file)}:")
                print(f"    Size: {result['size']}, Mode: {result['mode']}")
                print(f"    Mean: {result['mean']:.2f}, Std: {result['std']:.2f}")
                print(f"    Range: [{result['min']}, {result['max']}]")
                print(f"    Unique colors: {result['unique_colors']}")
                print(f"    Status: {result['status']}")
        
        # Check all images for blank/black
        blank_count = 0
        total_count = len(image_files)
        for img_file in image_files:
            result = analyze_image(img_file)
            if 'error' not in result and result['status'] == "BLANK/BLACK":
                blank_count += 1
        
        if blank_count > 0:
            print(f"\n  WARNING: {blank_count}/{total_count} images are blank/black ({blank_count/total_count*100:.1f}%)")
        else:
            print(f"\n  All {total_count} images appear to have content")

if __name__ == "__main__":
    compare_replicas()
