import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from sklearn import manifold
import os
from tqdm import tqdm

###################################################### for  pillar  feature ##########################################################
count = True
pillar_feature_name = "pillar_feature_rank/"

for idx in os.listdir("C:/Users/kk/Desktop/AutonomousCar/PointPillar/feature_distribution/" + pillar_feature_name):
    print(idx)
# idx = 140
    pillar_feature = np.load("C:/Users/kk/Desktop/AutonomousCar/PointPillar/feature_distribution/" + pillar_feature_name + idx)
    # print(pillar_feature.shape)
    delete_index = []
    for index in range(pillar_feature.shape[0]):
        if np.max(pillar_feature[index,0:64]) == 0:
        # if np.max(pillar_feature[index,0:128]) == 0:
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
    

X = pillar_features[:,0:64].copy()
# X = pillar_features[:,0:128].copy()

#t-SNE
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)

#Data Visualization
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize


y = pillar_features[:,65].astype("float32").copy()
# y = pillar_features[:,129].copy()

# for row in range(y.shape[0]):
#     if y[row] <= 100:
#         y[row] = 0
#     elif (y[row] > 100) & (y[row] <= 200):
#         y[row] = 1
#     elif (y[row] > 200) & (y[row] <= 300):
#         y[row] = 2
#     else:
#         y[row] = 3

# # y = pillar_features[:,64].copy()*pillar_features[:,65].copy()
# y = pillar_features[:,128].copy()*pillar_features[:,129].copy()

# for row in range(y.shape[0]):
#     if y[row] == 0:
#         y[row] = 0
#     elif (y[row] > 0) & (y[row] <= 150):
#         y[row] = 1
#     elif (y[row] > 150) & (y[row] <= 300):
#         y[row] = 2
#     else:
#         y[row] = 3
        

###################################################### for  pillar  feature ##########################################################
        
###################################################### for  point  feature ##########################################################

# list_ = os.listdir("C:/Users/kk/Desktop/output_result/")
# pillar_features = np.fromfile("C:/Users/kk/Desktop/output_result/" + list_[5], dtype=np.float32).reshape(-1, 132)

# X = pillar_features[:,3:130].copy()

# #t-SNE
# X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X)

# #Data Visualization
# x_min, x_max = X_tsne.min(0), X_tsne.max(0)
# X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize


# y = pillar_features[:,2].copy()

# for row in range(y.shape[0]):
#     if y[row] <= 20:
#         y[row] = 0
#     elif (y[row] > 20) & (y[row] <= 40):
#         y[row] = 1
#     elif (y[row] > 40) & (y[row] <= 60):
#         y[row] = 2
#     else:
#         y[row] = 3
###################################################### for  point  feature ##########################################################


###################################################### draw by quantize ##########################################################
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

# #保存图片本地
# # plt.savefig('power.png', dpi=300)  
# plt.show()
###################################################### draw by quantize ##########################################################

###################################################### draw by matplotlib cm ##########################################################

colors = cm.jet(np.linspace(0, 1, 432))
y_color = np.zeros((y.shape[0], 4))
for i in range(y.shape[0]):
    y_color[i, :] = colors[int(y[i]), :]
for position, color in tqdm(zip(X_norm,y_color)):
    fd = plt.scatter(position[0], position[1], s=30, color=color)

plt.colorbar(fd)
plt.show()

###################################################### draw by matplotlib cm ##########################################################