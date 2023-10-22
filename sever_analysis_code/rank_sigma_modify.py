import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from tsnecuda import TSNE
import os
from tqdm import tqdm
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
###################################################### for  pillar  feature ##########################################################
count = True
pillar_feature_name = "pillar_feature_2_original/"

break_time = 0

for idx in sorted(os.listdir("/data2/chihjen/second.pytorch/second/pytorch/" + pillar_feature_name)):
    print(idx)
    break_time += 1
    pillar_feature = np.load("/data2/chihjen/second.pytorch/second/pytorch/" + pillar_feature_name + idx)
            
    your_idx = np.array([idx])
    your_idx = np.repeat(your_idx, pillar_feature.shape[0]).reshape((pillar_feature.shape[0],1))
    pillar_feature = np.concatenate((pillar_feature,your_idx), axis=1)
    if count:
        pillar_features = pillar_feature.copy()
        count = False
        continue
    
    pillar_features = np.concatenate((pillar_features,pillar_feature), axis=0)
    
    if break_time == 100:
        break

pillar_feature_2 = pillar_features.copy()


count = True
pillar_feature_name = "pillar_feature_3_original/"

break_time = 0

for idx in sorted(os.listdir("/data2/chihjen/second.pytorch/second/pytorch/" + pillar_feature_name)):
    print(idx)
    break_time += 1
    pillar_feature = np.load("/data2/chihjen/second.pytorch/second/pytorch/" + pillar_feature_name + idx)
            
    your_idx = np.array([idx])
    your_idx = np.repeat(your_idx, pillar_feature.shape[0]).reshape((pillar_feature.shape[0],1))
    pillar_feature = np.concatenate((pillar_feature,your_idx), axis=1)
    if count:
        pillar_features = pillar_feature.copy()
        count = False
        continue
    
    pillar_features = np.concatenate((pillar_features,pillar_feature), axis=0)
    
    if break_time == 100:
        break

pillar_feature_3 = pillar_features.copy()
    
    
count = True
pillar_feature_name = "pillar_feature_d_original/"

break_time = 0

for idx in sorted(os.listdir("/data2/chihjen/second.pytorch/second/pytorch/" + pillar_feature_name)):
    print(idx)
    break_time += 1
    pillar_feature = np.load("/data2/chihjen/second.pytorch/second/pytorch/" + pillar_feature_name + idx)

    your_idx = np.array([idx])
    your_idx = np.repeat(your_idx, pillar_feature.shape[0]).reshape((pillar_feature.shape[0],1))
    pillar_feature = np.concatenate((pillar_feature,your_idx), axis=1)
    if count:
        pillar_features = pillar_feature.copy()
        count = False
        continue
    
    pillar_features = np.concatenate((pillar_features,pillar_feature), axis=0)
    
    if break_time == 100:
        break
    
pillar_features = np.concatenate((pillar_feature_2[:,0:128], pillar_feature_3[:,0:128], pillar_features), axis=1)


# X = pillar_features[:,0:64].copy()
# X = pillar_features[:,0:128].copy()
X = pillar_features[:,0:384].copy()


# eigenvalue change
# U, S, V = torch.svd(torch.from_numpy(X.astype("float32")))
# S[0:3] = 0
# X = torch.matmul(U, torch.matmul(torch.diag(S), V.transpose(-2, -1))).numpy()


#t-SNE
print("\n")
print("========================================== Start TSNE ==========================================")
# X_tsne = TSNE(n_components=2, perplexity=40, learning_rate=10).fit_transform(X)
X_tsne = TSNE(n_components=2, perplexity=75).fit_transform(X)
print("========================================== End TSNE ==========================================")
print("\n")

#Data Visualization
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize


# y = pillar_features[:,65].astype("float32").copy()
y = pillar_features[:,385].astype("float32").copy()
# y = pillar_features[:,129].astype("float32").copy()
# for row in range(y.shape[0]):
#     if y[row] <= 100:
#         y[row] = 0
#     elif (y[row] > 100) & (y[row] <= 200):
#         y[row] = 1
#     elif (y[row] > 200) & (y[row] <= 300):
#         y[row] = 2
#     else:
#         y[row] = 3
for row in range(y.shape[0]):
    if y[row] <= 125:
        y[row] = 0
    else:
        y[row] = 2

###################################################### draw by quantize ##########################################################
plt.rc('font', family='SimHei', size=8)
plt.rcParams['axes.unicode_minus']=False 

flag = (y == 0)
X_norms = X_norm[flag]

L1 = [n[0] for n in X_norms]
L2 = [n[1] for n in X_norms]
 
plt.scatter(L1,L2,s=30,c='red',marker="+")

flag = (y == 1)
X_norms = X_norm[flag]

L1 = [n[0] for n in X_norms]
L2 = [n[1] for n in X_norms]

plt.scatter(L1,L2,s=30,c='blue',marker="x") 


flag = (y == 2)
X_norms = X_norm[flag]

L1 = [n[0] for n in X_norms]
L2 = [n[1] for n in X_norms]


plt.scatter(L1,L2,s=30,c='green',marker="*") 

flag = (y == 3)
X_norms = X_norm[flag]

L1 = [n[0] for n in X_norms]
L2 = [n[1] for n in X_norms]


plt.scatter(L1,L2,s=30,c='yellow',marker=">") 

#保存图片本地
# plt.savefig('power.png', dpi=300)  
plt.show()
###################################################### draw by quantize ##########################################################
