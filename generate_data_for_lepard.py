import argparse
import numpy as np
import os
import pickle as pkl
from plyfile import PlyData
import torch

def read_ply_file(filename):
    if not filename.endswith(".ply"):
        raise ValueError("Works with ply files only.")
    data = PlyData.read(filename)
    data = np.asarray([[float(x) for x in t] for t in data["vertex"].data])
    return data

def synthesize_and_save_4dmatch_dataset(s_pc, t_pc, path):
    if not path.endswith(".npz"):
        raise ValueError("Need to save in npz format for 4DMatch lepard.")
    np.savez(path,
        s_pc=s_pc, t_pc=t_pc, s2t_flow=np.zeros_like(s_pc), correspondences=np.asarray([[]]),
        s_overlap_rate=np.asarray(0), rot=np.zeros([3,3]), trans=np.zeros([3,1]),
        metric_index=np.asarray([[]]))

def synthesize_and_save_3dmatch_dataset(s_pc, t_pc, pth_dir, pkl_dir):
    s_pc_path = os.path.join(pth_dir, "source_point_cloud.pth")
    t_pc_path = os.path.join(pth_dir, "target_point_cloud.pth")
    torch.save(s_pc, s_pc_path)
    torch.save(t_pc, t_pc_path)
    pkl_path = os.path.join(pkl_dir, "processed_inputs.pkl")
    dict_for_pkl = {
        "rot": [np.zeros([3, 3])],
        "trans": [np.zeros([3, 1])],
        "src": [s_pc_path,],
        "tgt": [t_pc_path,],
        "overlap": [0,],
        "gt_cov": [np.asarray([[]])]
        }
    with open(pkl_path, "wb") as f:
        pkl.dump(dict_for_pkl, f)
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", help="Registration Mode - (rigid/non-rigid)")
    parser.add_argument("-s", "--src_path", help="Path to source point cloud")
    parser.add_argument("-t", "--tgt_path", help="Path to target point cloud")
    parser.add_argument("-o", "--out_dir_path", help="Path to processed data directory.")
    args = parser.parse_args()
    if args.mode is None:
        mode = input("Please enter the mode: (3DMatch/4DMatch) ")
        if mode not in ["rigid", "non-rigid"]:
            raise ValueError(f"Invalid Mode {mode}. Allowed values: rigid, non-rigid")
    else:
        mode = args.mode
    if args.src_path is None:
        source_point_cloud_path = input("Please enter path to source point cloud: ")
    else:
        source_point_cloud_path = args.src_path

    s_pc = read_ply_file(source_point_cloud_path)

    print("Source point cloud shape: ", s_pc.shape)

    if args.tgt_path is None:
        target_point_cloud_path = input("Please enter path to target point cloud: ")
    else:
        target_point_cloud_path = args.tgt_path

    t_pc = read_ply_file(target_point_cloud_path)

    print("Target point cloud shape: ", t_pc.shape)

    if args.out_dir_path is None:
        output_dir_path = input("Please enter the path of the output directory: ")
    else:
        output_dir_path = args.out_dir_path

    if mode == "non-rigid":
        npz_file_path = os.path.join(output_dir_path, "processed_inputs.npz")
        synthesize_and_save_4dmatch_dataset(s_pc=s_pc, t_pc=t_pc, path=npz_file_path)
        print(f"Successfully exported data in 4DMatch format and saved in {npz_file_path}. Best of luck with registration!")
    elif mode == "rigid":
        synthesize_and_save_3dmatch_dataset(s_pc=s_pc, t_pc=t_pc, pth_dir=output_dir_path, pkl_dir=output_dir_path)
        print(f"Successfully exported data in 3DMatch format and saved in {output_dir_path}. Best of luck with registration!")


    
