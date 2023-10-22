import matplotlib.pyplot as plt
import numpy as np

# for idx in range(7481):
#     print(idx)
idx = 6258
a = np.squeeze(np.load("C:/Users/kk/Desktop/label_grid_num_person/" + "%06d.npy" % idx))
a[a<0] = 0
plt.figure("heat map")
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.clim(0, np.max(a))
plt.colorbar()
plt.show()
# plt.savefig("C:/Users/kk/Desktop/density_heat_map/" + "%06d.png" % idx)
# plt.close()
