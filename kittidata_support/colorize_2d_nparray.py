import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
from matplotlib.colors import ListedColormap

idx = 427

# predict_combine_ratio1 = np.rot90(np.load("C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_grid_num/"  + "%06d.npy" % idx).transpose(2,1,0)[0,::-1,::-1])
predict_combine_ratio1 = np.rot90(np.load("C:/Users/kk/Desktop/AutonomousCar/PointPillar/combine_ratio_7/"  + "%06d.npy" % idx)[::-1,:])

# cmap = ListedColormap(["#0000FF", "#0020FF", "#0040FF", "#0060FF", "#0080FF", "#00A0FF", "#00C0FF", "#00FFFF", "#00FFC0", "#00FF80", "#00FF40", "#00FF00", "#40FF00", "#80FF00", "#C0FF00", "#FFFF00", "#FFC000", "#FFA000", "#FF8000", "#FF6000", "#FF4000", "#FF2000", "#FF0000"])

name = "jet"
plt.imshow(predict_combine_ratio1, cmap=plt.get_cmap(name)) 

# plt.imshow(np.sqrt(predict_combine_ratio1 - 1), cmap=plt.get_cmap(name)) 
# plt.imshow(np.sqrt(predict_combine_ratio1 - 1)[170:,200:300] + 1, cmap=cmap) 

plt.colorbar()


# save .mat file

# save_fn = 'C:/Users/kk/Desktop/p1.mat'
# save_array = predict_combine_ratio1
# sio.savemat(save_fn, {'array': save_array})


