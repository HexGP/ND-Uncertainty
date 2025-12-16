"""
Diagnostic script to investigate scan 3 training failure and uncertainty visualization issues.
"""
import os
import numpy as np
import trimesh
from pathlib import Path
import glob

def check_scan3_meshes():
    """Check mesh files for scan 3 to identify when/why training failed."""
    runs_dir = "runs_new"
    scan3_dir = os.path.join(runs_dir, "replica_3")
    
    if not os.path.exists(scan3_dir):
        print(f"ERROR: {scan3_dir} does not exist!")
        return
    
    # Find all experiment directories for scan 3
    exp_dirs = [d for d in os.listdir(scan3_dir) if os.path.isdir(os.path.join(scan3_dir, d))]
    if not exp_dirs:
        print(f"ERROR: No experiment directories found in {scan3_dir}")
        return
    
    print(f"Found {len(exp_dirs)} experiment directory(ies) for scan 3\n")
    
    for exp_dir in sorted(exp_dirs):
        exp_path = os.path.join(scan3_dir, exp_dir)
        plots_dir = os.path.join(exp_path, "plots")
        
        if not os.path.exists(plots_dir):
            print(f"  {exp_dir}: No plots/ directory found")
            continue
        
        # Find all mesh files
        mesh_files = sorted(glob.glob(os.path.join(plots_dir, "mesh_*.ply")))
        
        if not mesh_files:
            print(f"  {exp_dir}: No mesh files found")
            continue
        
        print(f"\n=== {exp_dir} ===")
        print(f"Found {len(mesh_files)} mesh file(s):")
        
        valid_meshes = []
        corrupted_meshes = []
        
        for mesh_file in mesh_files:
            mesh_name = os.path.basename(mesh_file)
            size = os.path.getsize(mesh_file)
            
            # Try to load the mesh
            try:
                mesh = trimesh.load(mesh_file, process=False)
                
                if isinstance(mesh, trimesh.Trimesh):
                    num_vertices = len(mesh.vertices)
                    num_faces = len(mesh.faces)
                    status = "VALID"
                    valid_meshes.append((mesh_name, size, num_vertices, num_faces))
                elif isinstance(mesh, trimesh.Scene):
                    num_geoms = len(mesh.geometry)
                    status = f"SCENE (geometries: {num_geoms})"
                    if num_geoms == 0:
                        corrupted_meshes.append((mesh_name, size, "Empty Scene"))
                    else:
                        valid_meshes.append((mesh_name, size, f"Scene with {num_geoms} geometries", None))
                else:
                    status = "UNKNOWN TYPE"
                    corrupted_meshes.append((mesh_name, size, status))
            except Exception as e:
                status = f"ERROR: {str(e)[:50]}"
                corrupted_meshes.append((mesh_name, size, status))
            
            print(f"  {mesh_name}: {size/1024:.2f} KB - {status}")
            if isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, 'vertices'):
                print(f"    Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
        
        # Check for training logs to see when it stopped
        log_dir = os.path.join(exp_path, "logs")
        if os.path.exists(log_dir):
            log_files = glob.glob(os.path.join(log_dir, "*.txt"))
            if log_files:
                print(f"\n  Log files found: {len(log_files)}")
                for log_file in sorted(log_files):
                    size = os.path.getsize(log_file)
                    print(f"    {os.path.basename(log_file)}: {size/1024:.2f} KB")
        
        # Summary
        print(f"\n  Summary: {len(valid_meshes)} valid, {len(corrupted_meshes)} corrupted/empty")
        
        if corrupted_meshes:
            print(f"\n  CORRUPTED/EMPTY MESHES:")
            for name, size, reason in corrupted_meshes:
                print(f"    {name}: {size} bytes - {reason}")


def check_uncertainty_images():
    """Check uncertainty visualization images to understand the lime green issue."""
    runs_dir = "runs_new"
    
    print("\n" + "="*60)
    print("CHECKING UNCERTAINTY VISUALIZATION IMAGES")
    print("="*60 + "\n")
    
    # Check multiple scans
    for scan_id in [1, 2, 3, 4]:
        scan_dir = os.path.join(runs_dir, f"replica_{scan_id}")
        if not os.path.exists(scan_dir):
            continue
        
        # Find latest experiment
        exp_dirs = sorted([d for d in os.listdir(scan_dir) if os.path.isdir(os.path.join(scan_dir, d))])
        if not exp_dirs:
            continue
        
        latest_exp = exp_dirs[-1]
        uncertainty_dir = os.path.join(scan_dir, latest_exp, "plots", "uncertainty")
        
        if not os.path.exists(uncertainty_dir):
            print(f"Scan {scan_id}: No uncertainty/ directory")
            continue
        
        # Get uncertainty images
        uncertainty_files = sorted(glob.glob(os.path.join(uncertainty_dir, "*.png")))
        
        if not uncertainty_files:
            print(f"Scan {scan_id}: No uncertainty images found")
            continue
        
        print(f"\nScan {scan_id} ({latest_exp}):")
        print(f"  Found {len(uncertainty_files)} uncertainty image(s)")
        
        # Check a few images
        for img_file in uncertainty_files[:3]:  # Check first 3
            try:
                from PIL import Image
                import numpy as np
                
                img = Image.open(img_file)
                img_array = np.array(img)
                
                # Check if image is uniform (all same color)
                unique_colors = len(np.unique(img_array.reshape(-1, img_array.shape[-1]), axis=0))
                mean_color = img_array.mean(axis=(0, 1))
                std_color = img_array.std(axis=(0, 1))
                
                img_name = os.path.basename(img_file)
                print(f"    {img_name}:")
                print(f"      Unique colors: {unique_colors}")
                print(f"      Mean RGB: ({mean_color[0]:.1f}, {mean_color[1]:.1f}, {mean_color[2]:.1f})")
                print(f"      Std RGB: ({std_color[0]:.2f}, {std_color[1]:.2f}, {std_color[2]:.2f})")
                
                if unique_colors < 10:
                    print(f"      WARNING: Image appears uniform (only {unique_colors} unique colors)")
                if std_color.max() < 1.0:
                    print(f"      WARNING: Very low variance (std < 1), image is nearly uniform")
                    
            except Exception as e:
                print(f"    ERROR reading {img_file}: {e}")


def check_scan3_data():
    """Check if scan 3 data exists and is accessible."""
    print("\n" + "="*60)
    print("CHECKING SCAN 3 DATA FILES")
    print("="*60 + "\n")
    
    data_dir = "data/Replica"
    scan3_data = os.path.join(data_dir, "scan3")
    
    required_files = [
        "traj.txt",
        "cameras.npz",
        "images",
        "depths",
        "normals"
    ]
    
    print(f"Checking data directory: {scan3_data}")
    
    if not os.path.exists(scan3_data):
        print(f"ERROR: {scan3_data} does not exist!")
        return False
    
    all_exist = True
    for req_file in required_files:
        file_path = os.path.join(scan3_data, req_file)
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"  {status} {req_file}")
        if not exists:
            all_exist = False
    
    if all_exist:
        # Check image count
        images_dir = os.path.join(scan3_data, "images")
        if os.path.exists(images_dir):
            img_files = glob.glob(os.path.join(images_dir, "*.jpg")) + \
                       glob.glob(os.path.join(images_dir, "*.png"))
            print(f"\n  Found {len(img_files)} image files")
    
    return all_exist


if __name__ == "__main__":
    print("="*60)
    print("SCAN 3 DIAGNOSTIC SCRIPT")
    print("="*60)
    
    # 1. Check scan 3 mesh files
    print("\n1. CHECKING SCAN 3 MESH FILES")
    print("-"*60)
    check_scan3_meshes()
    
    # 2. Check uncertainty images
    check_uncertainty_images()
    
    # 3. Check scan 3 data
    data_ok = check_scan3_data()
    
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    print("\nPossible issues identified:")
    print("1. If meshes are corrupted/empty → training collapsed (check for NaN/OOM errors)")
    print("2. If uncertainty images are uniform → all beta values clamped to sigma_max=0.5")
    print("3. If scan 3 data missing → cannot train or evaluate scan 3")
    print("\nRecommendations:")
    print("- Check training logs for errors around epoch 2016")
    print("- Verify sigma clamping is working correctly in loss_wrapper.py")
    print("- Consider re-training scan 3 with fixed code")
    print("="*60)
