from tqdm import tqdm
import numpy as np
import os

path = "/data2/chihjen/second.pytorch/second/cls_acc_DAWN_RANK/"
threshold = 0.007

distance = 2

first = True

data_list = os.listdir(path)
for i in tqdm(data_list):
    cur_sample = np.load(path + i)
    cur_gt = cur_sample[:,0]
    cur_p = cur_sample[:,1]
    cur_d = cur_sample[:,2]
    
    if distance == 0:
        cur_flag_d = cur_d<63
    elif distance == 1:
        cur_flag_d = (cur_d>=63) * (cur_d<125)
    elif distance == 2:
        cur_flag_d = cur_d>=125
        
    
    cur_flag = cur_gt==1
    
    if distance == 0 or distance == 1 or distance == 2:
        cur_p = cur_p[cur_flag*cur_flag_d]
    else:
        cur_p = cur_p[cur_flag]
    
    if first:
        acc_p = cur_p.copy()
        first = False
    else:
        acc_p = np.concatenate((acc_p, cur_p))

acc_p_score = (acc_p > threshold)*1
acc = np.sum(acc_p_score)/acc_p_score.shape
print("\n")
print(acc)