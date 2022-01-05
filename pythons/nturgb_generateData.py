import glob
import os, sys
import numpy as np
import csv

NPY_DIR = '../../Datasets/5_nturgb+d/nturgb+d_npy/'
frame_dict = {}
dataset = np.array([])
fileset = np.array([])
seq_no_arr = [20, 30, 40, 50, 60]

for idx, file_name in enumerate(glob.glob(os.path.join(NPY_DIR, '*.npy'))):
    file = np.load(file_name)
    frame_dict[file_name] = file.shape[0]    

for seq_no in seq_no_arr:
    for k, v in frame_dict.items():
        if v > seq_no:
            frameset = np.array([])
            fileset = np.array([])
            file = np.load(k)
            for frame in range(int(v/seq_no)*seq_no):
                if frameset.shape[0] == 0:
                    frameset = file[frame].reshape(file.shape[1], 1, file.shape[2])
                else:
                    frameset = np.concatenate((frameset, file[frame].reshape(file.shape[1], 1, file.shape[2])), axis=1)
                    if (frameset.shape[1]) % seq_no == 0:
                        if fileset.shape[0] == 0:
                            fileset = frameset.reshape(frameset.shape[0], frameset.shape[1]*frameset.shape[2])
                        else:
                            fileset = np.concatenate((fileset, frameset.reshape(frameset.shape[0], frameset.shape[1]*frameset.shape[2])), axis=0)
                        frameset = np.array([])
                    else:
                        frameset = np.concatenate((frameset, file[frame].reshape(file.shape[1], 1, file.shape[2])), axis=1)
            fileset = np.concatenate((fileset, np.tile(int(k.split('.')[-2].split('A')[-1]), fileset.shape[0]).reshape(-1, 1)), axis=1)
            dataset = fileset if dataset.shape[0] == 0 else np.concatenate((dataset, fileset), axis=0)
            display(dataset.shape)

    with open(f"../../Datasets/5_nturgb+d/nturgb+d.ni={3}.no={60}.ts={seq_no}.bs={50}.csv",'w') as csvfile:
        np.savetxt(csvfile, np.array([[3, 60]]),fmt='%d', delimiter=",")
    with open(f"../../Datasets/5_nturgb+d/nturgb+d.ni={3}.no={60}.ts={seq_no}.bs={50}.csv",'a') as csvfile:
        np.savetxt(csvfile, testset_np, fmt='%.4f', delimiter=",")