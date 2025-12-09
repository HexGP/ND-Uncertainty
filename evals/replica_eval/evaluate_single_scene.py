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

    out_dir = args.output_dir
    data_dir = args.data_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    idx = args.scan_id
    scan = scans[int(idx) - 1]

    ply_file = args.input_mesh

    result_mesh_file = os.path.join(out_dir, "culled_mesh_.ply")

    # cumesh
    cull_mesh_out = os.path.join(out_dir, f"cull_{scan}.ply")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cmd = f"python {os.path.join(script_dir, 'cull_mesh.py')} --input_mesh {ply_file} --input_scalemat {data_dir}/scan{idx}/cameras.npz --traj {data_dir}/scan{idx}/traj.txt --output_mesh {cull_mesh_out}"
    print(cmd)
    os.system(cmd)

    gt_mesh_path = os.path.join(data_dir, 'cull_GTmesh', f"{scan}.ply")
    gt_mesh = trimesh.load(gt_mesh_path)
    gt_mesh.export(os.path.join(out_dir, f"{scan}_gt.ply"))
    cmd = f"python {os.path.join(script_dir, 'eval_recon.py')} --rec_mesh {cull_mesh_out} --gt_mesh {gt_mesh_path}"
    print(cmd)
    # accuracy_rec, completion_rec, precision_ratio_rec, completion_ratio_rec, fscore, normal_acc, normal_comp, normal_avg
    output = subprocess.check_output(cmd, shell=True).decode("utf-8")
    output = output.replace(" ", ",")
    print(output)
