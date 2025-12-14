import argparse
import os
import glob
import re
from datetime import datetime

import trimesh
from pathlib import Path
import subprocess

def fix_mesh_if_scene(mesh_file, out_dir, scan_name):
    """If mesh file is a Scene (multiple meshes), convert to single mesh and save."""
    try:
        mesh = trimesh.load(mesh_file, process=False)
        if isinstance(mesh, trimesh.Trimesh):
            return mesh_file
        
        if isinstance(mesh, trimesh.Scene):
            print(f"  Converting Scene to single mesh...")
            meshes_to_combine = []
            for key in mesh.geometry.keys():
                geom = mesh.geometry[key]
                if isinstance(geom, trimesh.Trimesh):
                    meshes_to_combine.append(geom)
            
            if not meshes_to_combine:
                print(f"  Warning: Scene has no valid meshes, trying manual extraction...")
                # Manual extraction
                import numpy as np
                all_verts = []
                all_faces = []
                offset = 0
                for name, geom in mesh.geometry.items():
                    if hasattr(geom, 'vertices') and hasattr(geom, 'faces'):
                        verts = np.array(geom.vertices)
                        faces = np.array(geom.faces)
                        if len(verts) > 0 and len(faces) > 0:
                            all_verts.append(verts)
                            all_faces.append(faces + offset)
                            offset += len(verts)
                if all_verts:
                    combined_verts = np.vstack(all_verts)
                    combined_faces = np.vstack(all_faces)
                    single_mesh = trimesh.Trimesh(vertices=combined_verts, faces=combined_faces)
                    fixed_file = os.path.join(out_dir, f"{scan_name}_fixed.ply")
                    single_mesh.export(fixed_file)
                    print(f"  Created fixed mesh: {fixed_file}")
                    return fixed_file
                return mesh_file
            
            # Combine meshes
            try:
                combined_mesh = trimesh.util.concatenate(meshes_to_combine)
            except:
                # Manual combination
                import numpy as np
                all_verts = []
                all_faces = []
                offset = 0
                for m in meshes_to_combine:
                    all_verts.append(m.vertices)
                    all_faces.append(m.faces + offset)
                    offset += len(m.vertices)
                combined_verts = np.vstack(all_verts)
                combined_faces = np.vstack(all_faces)
                combined_mesh = trimesh.Trimesh(vertices=combined_verts, faces=combined_faces)
            
            fixed_file = os.path.join(out_dir, f"{scan_name}_fixed.ply")
            combined_mesh.export(fixed_file)
            print(f"  Created fixed mesh: {fixed_file}")
            return fixed_file
        
        return mesh_file
    except Exception as e:
        print(f"  Warning: Could not fix Scene mesh: {e}, using original")
        return mesh_file

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default='../../runs')
parser.add_argument('--data_dir', type=str, default='../../data/Replica')
parser.add_argument('--exp_name', type=str, default='replica')
parser.add_argument('--out_dir', type=str, default='evaluation')
parser.add_argument('--use_latest', action='store_true', help='Use latest timestamp instead of first')
args = parser.parse_args()

scans = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
exp_scans = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]
data_dir = args.data_dir
root_dir = args.root_dir
exp_name = args.exp_name
out_dir = os.path.join(args.out_dir, exp_name)
Path(out_dir).mkdir(parents=True, exist_ok=True)

script_dir = os.path.dirname(os.path.abspath(__file__))

evaluation_txt_file = f"{args.out_dir}/{exp_name}.csv"
evaluation_txt_file = open(evaluation_txt_file, 'w')

for idx, scan in enumerate(scans):
    idx = idx + 1
    # test set
    if not (scan in exp_scans):
       continue

    cur_exp = f"{exp_name}_{idx}"
    cur_root = os.path.join(root_dir, cur_exp)
    
    if not os.path.exists(cur_root):
        print(f"Skipping {scan}: {cur_exp} not found")
        continue
    
    # Get timestamp directories
    dirs = [d for d in os.listdir(cur_root) if os.path.isdir(os.path.join(cur_root, d))]
    if not dirs:
        print(f"Skipping {scan}: No timestamp directories found in {cur_exp}")
        continue
    
    # Use latest or first timestamp
    if args.use_latest:
        # Sort by timestamp (format: YYYY-MM-DD_HH-MM-SS)
        dirs.sort(key=lambda x: datetime.strptime(x, '%Y-%m-%d_%H-%M-%S') if re.match(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}', x) else datetime.min)
        cur_root = os.path.join(cur_root, dirs[-1])
    else:
        # Use first (original behavior)
        dirs = sorted(dirs)
        cur_root = os.path.join(cur_root, dirs[0])
    
    files = list(filter(os.path.isfile, glob.glob(os.path.join(cur_root, "plots/*.ply"))))
    if not files:
        print(f"Skipping {scan}: No .ply files found in {cur_root}/plots/")
        continue

    files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]) if x.split('/')[-1].split('.')[0].split('_')[-1].isdigit() else 0)
    ply_file = files[-1]
    print(f"\nProcessing {scan} (scan{idx})...")
    print(f"  Using mesh: {ply_file}")

    # Fix Scene objects before culling
    fixed_mesh = fix_mesh_if_scene(ply_file, out_dir, scan)
    
    # Cull mesh
    cull_mesh_out = os.path.join(out_dir, f"cull_{scan}.ply")
    cameras_file = os.path.join(data_dir, f"scan{idx}/cameras.npz")
    traj_file = os.path.join(data_dir, f"scan{idx}/traj.txt")
    
    if os.path.exists(cameras_file) and os.path.exists(traj_file):
        cmd = f"python {os.path.join(script_dir, 'cull_mesh.py')} --input_mesh {fixed_mesh} --input_scalemat {cameras_file} --traj {traj_file} --output_mesh {cull_mesh_out}"
        print(f"  {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0 or not os.path.exists(cull_mesh_out):
            print(f"  Warning: Culling failed, using fixed mesh")
            cull_mesh_out = fixed_mesh
    else:
        print(f"  Warning: cameras.npz or traj.txt not found, skipping culling")
        cull_mesh_out = fixed_mesh

    # Evaluate
    gt_mesh = os.path.join(data_dir, f"cull_GTmesh/{scan}.ply")
    cmd = f"python {os.path.join(script_dir, 'eval_recon.py')} --rec_mesh {cull_mesh_out} --gt_mesh {gt_mesh}"
    print(f"  {cmd}")
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode("utf-8")
        output = output.replace(" ", ",")
        print(output)
        evaluation_txt_file.write(f"{scan},{Path(ply_file).name},{output}")
        evaluation_txt_file.flush()
    except subprocess.CalledProcessError as e:
        print(f"  Error: Evaluation failed: {e.output.decode('utf-8')[:200] if e.output else 'No output'}")
