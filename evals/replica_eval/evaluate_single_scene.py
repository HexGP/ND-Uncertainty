import os
from pathlib import Path
import subprocess
import argparse

import trimesh

scans = ["room0", "room1", "room2", "office0", "office1", "office2", "office3", "office4"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the mesh.'
    )
    parser.add_argument('--input_mesh', type=str, default='/home/dawn/replica_meshes/replica_explore11_6_mesh_2000.ply',help='path to the mesh to be evaluated')
    parser.add_argument('--scan_id', type=str, default='6',help='scan id of the input mesh')
    parser.add_argument('--data_dir', type=str, default='../../data/Replica', help='path to the dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation_results_single', help='path to the output folder')
    args = parser.parse_args()

    # Convert relative paths to absolute paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))  # Go up from evals/replica_eval to ND-Uncertainty
    
    # Resolve paths: if relative, make them relative to project root, then convert to absolute
    if os.path.isabs(args.output_dir):
        out_dir = args.output_dir
    elif args.output_dir.startswith('../'):
        out_dir = os.path.abspath(args.output_dir)
    else:
        out_dir = os.path.abspath(os.path.join(project_root, args.output_dir))
    
    if os.path.isabs(args.data_dir):
        data_dir = args.data_dir
    elif args.data_dir.startswith('../'):
        data_dir = os.path.abspath(args.data_dir)
    else:
        data_dir = os.path.abspath(os.path.join(project_root, args.data_dir))
    
    if os.path.isabs(args.input_mesh):
        ply_file = args.input_mesh
    elif args.input_mesh.startswith('../'):
        ply_file = os.path.abspath(args.input_mesh)
    else:
        ply_file = os.path.abspath(os.path.join(project_root, args.input_mesh))
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    idx = args.scan_id
    scan = scans[int(idx) - 1]

    result_mesh_file = os.path.join(out_dir, "culled_mesh_.ply")

    # cumesh
    cull_mesh_out = os.path.join(out_dir, f"cull_{scan}.ply")
    cameras_file = os.path.join(data_dir, f'scan{idx}', 'cameras.npz')
    traj_file = os.path.join(data_dir, f'scan{idx}', 'traj.txt')
    
    # Check if required files exist
    if not os.path.exists(cameras_file):
        print(f"Warning: cameras.npz not found at {cameras_file}, skipping culling")
        cull_mesh_out = ply_file  # Use original mesh
    elif not os.path.exists(traj_file):
        print(f"Warning: traj.txt not found at {traj_file}, skipping culling")
        cull_mesh_out = ply_file  # Use original mesh
    else:
        cmd = f"python {os.path.join(script_dir, 'cull_mesh.py')} --input_mesh {ply_file} --input_scalemat {cameras_file} --traj {traj_file} --output_mesh {cull_mesh_out}"
        print(cmd)
        result = os.system(cmd)
        if result != 0 or not os.path.exists(cull_mesh_out):
            print(f"Warning: Culling failed, using original mesh")
            cull_mesh_out = ply_file

    gt_mesh_path = os.path.join(data_dir, 'cull_GTmesh', f"{scan}.ply")
    if not os.path.exists(gt_mesh_path):
        print(f"Error: GT mesh not found at {gt_mesh_path}")
        exit(1)
    gt_mesh = trimesh.load(gt_mesh_path)
    gt_mesh.export(os.path.join(out_dir, f"{scan}_gt.ply"))
    cmd = f"python {os.path.join(script_dir, 'eval_recon.py')} --rec_mesh {cull_mesh_out} --gt_mesh {gt_mesh_path}"
    print(cmd)
    # accuracy_rec, completion_rec, precision_ratio_rec, completion_ratio_rec, fscore, normal_acc, normal_comp, normal_avg
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    output = output.replace(" ", ",")
    print(output)
