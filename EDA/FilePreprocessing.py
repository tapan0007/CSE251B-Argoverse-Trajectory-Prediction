import pickle
import os
from tqdm import tqdm
from glob import glob
from random import sample
import numpy as np

train_data_path = "./train/train"
train_pkl_list = glob(os.path.join(train_data_path, '*'))
# train_pkl_list.sort()
# train_pkl_list = sample(train_pkl_list, int(0.45*len(train_pkl_list)))


out_directory_path = "./train_tracked_agent"
if not os.path.exists(out_directory_path):
    os.mkdir(out_directory_path)
for pkl_path in tqdm(train_pkl_list):
    with open(pkl_path, 'rb') as f:
        scene = pickle.load(f)
        # the index of agent to be predicted 
        pred_id = np.where(scene["track_id"] == scene['agent_id'])[0][0]
        
        # input: p_in & v_in; output: p_out
        inp_scene = np.dstack([scene['p_in'], scene['v_in']])
        out_scene = np.dstack([scene['p_out'], scene['v_out']])
        
        # Normalization 
        min_vecs = np.min(inp_scene, axis = (0,1))
        max_vecs = np.max(inp_scene, axis = (0,1))
        
        # Normalize by vectors
        inp = (inp_scene[pred_id] - min_vecs)/(max_vecs - min_vecs)
        out = (out_scene[pred_id] - min_vecs)/(max_vecs - min_vecs)
        dataToWrite = {'inp': inp, 'out': out, 'city': scene['city']}

    # Store data (serialize)
    with open(os.path.join(out_directory_path, os.path.basename(pkl_path)), 'wb') as handle:
        pickle.dump(dataToWrite, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    #Check that outputs are equal
    # with open(os.path.join(out_directory_path, os.path.basename(pkl_path)), 'rb') as handle:
    #     # print(pickle.load(handle) == dataToWrite)
    #     print("Loaded pickle:")
    #     loaded = pickle.load(handle)
    #     print(loaded)
    #     print(type(loaded['inp']))
    #     print("Written data:")
    #     print(dataToWrite['inp'])
    #     print(type(dataToWrite))
    #     print(dataToWrite['inp'] == loaded['inp'])
    #     print(dataToWrite['out'] == loaded['out'])
    # break
