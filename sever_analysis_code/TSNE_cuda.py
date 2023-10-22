import numpy as np
import matplotlib.pyplot as plt
from tsnecuda import TSNE
import os
import random

def draw2D_quantize(X_norm, y, yy):
    
    # for row in range(y.shape[0]):
    #     if y[row] <= 60:
    #         y[row] = 0
    #     elif (y[row] > 60) & (y[row] <= 120):
    #         y[row] = 1
    #     elif (y[row] > 120) & (y[row] <= 180):
    #         y[row] = 2
    #     else:
    #         y[row] = 3
            
            
    for row in range(y.shape[0]):
        if yy[row] == 0:
            y[row] = 0
        else:
            y[row] = 2
    
    # for row in range(y.shape[0]):
    #     if yy[row] <= 10:
    #         y[row] = 0
    #     elif (yy[row] > 10) & (yy[row] <= 30):
    #         y[row] = 1
    #     else:
    #         y[row] = 2
    
    # for row in range(y.shape[0]):
    #     if yy[row] == 0:
    #         # y[row] = 0
    #         if y[row] > 250:
    #             y[row] = 0
    #         else:
    #             y[row] = -1
    #     else:
    #         if y[row] > 250:
    #             y[row] = 2
    #         else:
    #             y[row] = -1
    
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

def draw2D(X_norm, y):
    L1 = [n[0] for n in X_norm]
    L2 = [n[1] for n in X_norm]
    y_color = [n*0.16 for n in y]
    fd = plt.scatter(L1, L2, s=5, c=y_color, cmap="jet")
    
    plt.colorbar(fd)
    plt.show()
    
def draw3D(X_norm, y):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    L1 = [n[0] for n in X_norm]
    L2 = [n[1] for n in X_norm]
    L3 = [n[2] for n in X_norm]
    y_color = [n*0.16 for n in y]
    fd = ax.scatter(L1, L2, L3, s=5, c=y_color, cmap="jet")
    
    plt.colorbar(fd)
    
    ax.set_xlabel('PC0')
    ax.set_ylabel('PC1')
    ax.set_zlabel('PC2')
    
    plt.xlim([-0.03, 0.03])
    plt.ylim([-0.03, 0.03])
    ax.set_zlim([-0.03, 0.03])
    
    plt.show()

###################################################### for  pillar  feature ##########################################################
    
count = True
pillar_feature_name = "pillar_feature_rank_DAWN_catten_after_fgbg/"

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
    
    
    delete_index = []
    for index in range(pillar_feature.shape[0]):
        if pillar_feature[index, 384] == 0:
            if random.randrange(100) < 15:
                delete_index.append(index)
            
    pillar_feature = np.delete(pillar_feature,delete_index,axis = 0)
    
    
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


# X = pillar_features[:,0:64].astype("float32").copy()
# X = pillar_features[:,0:128].astype("float32").copy()
X = pillar_features[:,0:384].astype("float32").copy()
# X = pillar_features[:,0:128].astype("float32").copy()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
# yy = pillar_features[:,64].astype("float32").copy()
y = pillar_features[:,385].astype("float32").copy()
yy = pillar_features[:,384].astype("float32").copy()
# y = pillar_features[:,129].astype("float32").copy()
# yy = pillar_features[:,128].astype("float32").copy()


draw2D_quantize(X_norm, y, yy)
        
# draw2D(X_norm, y)

# draw3D(X_norm, y)