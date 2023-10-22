from matplotlib import pyplot as plt
import numpy as np
import torch
import os

count = True
pillar_feature_name = "pillar_feature_rank_DAWN_catten_nodf_before/"

break_time = 0

for idx in sorted(os.listdir("/data2/chihjen/second.pytorch/second/pytorch/" + pillar_feature_name)):
    print(break_time)
    break_time += 1
    pillar_feature = np.load("/data2/chihjen/second.pytorch/second/pytorch/" + pillar_feature_name + idx)
    
    # delete_index = []
    # for index in range(pillar_feature.shape[0]):
    #     if np.max(pillar_feature[index,0:64]) == 0:
    #         delete_index.append(index)
            
    # pillar_feature = np.delete(pillar_feature,delete_index,axis = 0)

    your_idx = np.array([idx])
    your_idx = np.repeat(your_idx, pillar_feature.shape[0]).reshape((pillar_feature.shape[0],1))
    pillar_feature = np.concatenate((pillar_feature,your_idx), axis=1)
    if count:
        before = pillar_feature.copy()
        count = False
        continue
    
    before = np.concatenate((before,pillar_feature), axis=0)
    
    if break_time == 100:
        break
    
count = True
pillar_feature_name = "pillar_feature_rank_DAWN_catten_nodf_after/"

break_time = 0

for idx in sorted(os.listdir("/data2/chihjen/second.pytorch/second/pytorch/" + pillar_feature_name)):
    print(break_time)
    break_time += 1
    pillar_feature = np.load("/data2/chihjen/second.pytorch/second/pytorch/" + pillar_feature_name + idx)
    
    # delete_index = []
    # for index in range(pillar_feature.shape[0]):
    #     if np.max(pillar_feature[index,0:64]) == 0:
    #         delete_index.append(index)
            
    # pillar_feature = np.delete(pillar_feature,delete_index,axis = 0)

    your_idx = np.array([idx])
    your_idx = np.repeat(your_idx, pillar_feature.shape[0]).reshape((pillar_feature.shape[0],1))
    pillar_feature = np.concatenate((pillar_feature,your_idx), axis=1)
    if count:
        after = pillar_feature.copy()
        count = False
        continue
    
    after = np.concatenate((after,pillar_feature), axis=0)
    
    if break_time == 100:
        break
    

select_threshold = 0.005

r_map = after[:,0:384].copy().astype("float32") - before[:,0:384].copy().astype("float32")    



b_map = (r_map < -select_threshold)*1
b_map = (r_map > select_threshold)*1 + b_map

# b_map = b_map[:, :128]
# b_map = b_map[:, 128:256]
b_map = b_map[:, 256:]

y = after[:,385].astype("float32").copy()

close = b_map[y<125]
mid = b_map[~((y<125)*1 + (y>=250)*1).astype("bool")]
far = b_map[y>=250]

close = np.mean(close, axis=0, keepdims=True)
mid = np.mean(mid, axis=0, keepdims=True)
far = np.mean(far, axis=0, keepdims=True)

result = np.concatenate((close,mid,far),axis=0)

plt.imshow(result, cmap=plt.get_cmap("gray"))


###########################################################################################################

# X = after[:,0:384].copy().astype("float32")
# X = X - np.mean(X, axis=0)
# U, S, V = torch.svd(torch.from_numpy(X))



# # r_map_channel_mean = np.mean(r_map, axis=0)
# # select_channle_1 = (r_map_channel_mean < -select_threshold)*1
# # select_channle_2 = (r_map_channel_mean > select_threshold)*1
# # select_channle = select_channle_1 + select_channle_2

# # for i in range(384):
# #     if(select_channle[i].astype("bool")):
# #         S[i] = 0
 
        
# # # # dynamic_channel = [2, 10, 30, 41, 43, 84, 92, 100]
# # # dynamic_channel = [10, 62, 75, 79, 85]
# # # for i in dynamic_channel:
# # #     # if(not (i in dynamic_channel)):
# # #     S[i] = 0

# # X = torch.matmul(U, torch.matmul(torch.diag(S), V.transpose(-2, -1))).numpy()
# # U, S, V = torch.svd(torch.from_numpy(X))




# u = U.numpy()

# X_norm = u[:,1:3].copy()

# y = after[:,385].astype("float32").copy()

# for row in range(y.shape[0]):
#     if y[row] <= 125:
#         y[row] = 0
#     else:
#         y[row] = 2
        
# plt.rc('font', family='SimHei', size=8)
# plt.rcParams['axes.unicode_minus']=False 

# flag = (y == 0)
# X_norms = X_norm[flag]

# L1 = [n[0] for n in X_norms]
# L2 = [n[1] for n in X_norms]
 
# plt.scatter(L1,L2,s=30,c='red',marker="+")

# flag = (y == 1)
# X_norms = X_norm[flag]

# L1 = [n[0] for n in X_norms]
# L2 = [n[1] for n in X_norms]

# plt.scatter(L1,L2,s=30,c='blue',marker="x") 


# flag = (y == 2)
# X_norms = X_norm[flag]

# L1 = [n[0] for n in X_norms]
# L2 = [n[1] for n in X_norms]


# plt.scatter(L1,L2,s=30,c='green',marker="*") 

# flag = (y == 3)
# X_norms = X_norm[flag]

# L1 = [n[0] for n in X_norms]
# L2 = [n[1] for n in X_norms]


# plt.scatter(L1,L2,s=30,c='yellow',marker=">") 


