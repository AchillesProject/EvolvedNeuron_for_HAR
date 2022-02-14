import glob
import os, sys
import numpy as np
import csv, math

SKELETON_DIR = '../../Datasets/5_nturgb+d/nturgb+d_skeletons'
NPY_DIR = '../../Datasets/5_nturgb+d/nturgb+d_npy/'
max_frame_count = 300
max_joints = 25
noIn = 3*max_joints
noOut = 60-10
seq_no_arr = [20, 30, 40, 50, 60]

def _read_skeleton(file_path, save_skelxyz=True, save_rgbxy=True, save_depthxy=True):
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the 
    # abundant bodys. 
    # read all lines into the pool to speed up, less io operation. 
    nframe = int(datas[0][:-1])
    bodymat = dict()
    bodymat['file_name'] = file_path[-29:-9]
    nbody = int(datas[1][:-1])
    bodymat['nbodys'] = [] 
    bodymat['njoints'] = njoints 
    for body in range(max_body):
        if save_skelxyz:
            bodymat['skel_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 3))
        if save_rgbxy:
            bodymat['rgb_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
        if save_depthxy:
            bodymat['depth_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])    
        if bodycount == 0:
            continue 
        # skip the empty frame 
        bodymat['nbodys'].append(bodycount)
        for body in range(bodycount):
            cursor += 1
            skel_body = 'skel_body{}'.format(body)
            rgb_body = 'rgb_body{}'.format(body)
            depth_body = 'depth_body{}'.format(body)
            
            bodyinfo = datas[cursor][:-1].split(' ')
            cursor += 1
            
            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                if save_skelxyz:
                    bodymat[skel_body][frame,joint] = jointinfo[:3]
                if save_depthxy:
                    bodymat[depth_body][frame,joint] = jointinfo[3:5]
                if save_rgbxy:
                    bodymat[rgb_body][frame,joint] = jointinfo[5:7]
    if len(bodymat['nbodys']) > 0:
        # prune the abundant bodys 
        for each in range(max_body):
            if each >= max(bodymat['nbodys']):
                if save_skelxyz:
                    del bodymat['skel_body{}'.format(each)]
                if save_rgbxy:
                    del bodymat['rgb_body{}'.format(each)]
                if save_depthxy:
                    del bodymat['depth_body{}'.format(each)]
        nbodies = max(bodymat['nbodys'])
    else:
        nbodies = 0
    return nbodies, nframe, bodymat 

if __name__ == '__main__':
    skeleton_files_mask = os.path.join(SKELETON_DIR, '*.skeleton')
    skeleton_files = sorted(glob.glob(skeleton_files_mask))

    for seq_no in seq_no_arr:
        dataset = np.array([])
        for idx, file_name in enumerate(skeleton_files):
            print(seq_no, file_name)
            fileset = np.array([])
            activity = int(file_name.split('.')[-2].split('A')[-1])
            activity_arr = np.zeros([noOut,1])
            if activity < 50:
                nbodies, nframe, mat = _read_skeleton(file_name, save_skelxyz=True, save_rgbxy=False, save_depthxy=False)
                if nframe >= seq_no and nbodies > 0:
                    activity_arr[int(activity)-1] = 1
                    for nbody in range(nbodies): 
                        skel_body = f'skel_body{nbody}'
                        if skel_body in mat:
                            for i in range(math.ceil(nframe/seq_no)):
                                if (i+1)*seq_no < nframe:
                                    if fileset.shape[0] == 0:
                                        fileset = mat[skel_body][i*seq_no:(i+1)*seq_no].reshape(1, seq_no, -1)
                                    else:
                                        fileset = np.concatenate((fileset, mat[skel_body][i*seq_no:(i+1)*seq_no].reshape(1, seq_no, -1)), axis=0)
                                else:
                                    diff_no = (i+1)*seq_no - nframe
                                    fileset = np.concatenate((fileset, mat[skel_body][i*seq_no-diff_no:(i+1)*seq_no-diff_no].reshape(1, seq_no, -1)), axis=0)
                fileset = fileset.reshape(fileset.shape[0],-1)  
                fileset = np.concatenate((fileset, np.tile(activity_arr, fileset.shape[0]).T), axis=1)
                dataset = fileset if dataset.shape[0] == 0 else np.concatenate((dataset, fileset), axis=0)
                print(dataset.shape)

        with open(f"../../Datasets/5_nturgb+d/nturgb+d.ni={noIn}.no={noOut}.ts={seq_no}.csv",'w') as csvfile:
            np.savetxt(csvfile, np.array([[noIn, noOut]]),fmt='%d', delimiter=",")
        with open(f"../../Datasets/5_nturgb+d/nturgb+d.ni={noIn}.no={noOut}.ts={seq_no}.csv",'a') as csvfile:
            np.savetxt(csvfile, dataset, fmt='%.4f', delimiter=",")
