import scipy.io as io
import numpy as np
import os

   
# count = True
# pillar_feature_name = "pillar_feature_2_DAWN/"

# break_time = 0

# for idx in sorted(os.listdir("/data2/chihjen/second.pytorch/second/pytorch/" + pillar_feature_name)):
#     print(idx)
#     break_time += 1
#     pillar_feature = np.load("/data2/chihjen/second.pytorch/second/pytorch/" + pillar_feature_name + idx)
            
#     your_idx = np.array([idx])
#     your_idx = np.repeat(your_idx, pillar_feature.shape[0]).reshape((pillar_feature.shape[0],1))
#     pillar_feature = np.concatenate((pillar_feature,your_idx), axis=1)
#     if count:
#         pillar_features = pillar_feature.copy()
#         count = False
#         continue
    
#     pillar_features = np.concatenate((pillar_features,pillar_feature), axis=0)
    
#     if break_time == 100:
#         break

# pillar_feature_2 = pillar_features.copy()


# count = True
# pillar_feature_name = "pillar_feature_3_DAWN/"

# break_time = 0

# for idx in sorted(os.listdir("/data2/chihjen/second.pytorch/second/pytorch/" + pillar_feature_name)):
#     print(idx)
#     break_time += 1
#     pillar_feature = np.load("/data2/chihjen/second.pytorch/second/pytorch/" + pillar_feature_name + idx)
            
#     your_idx = np.array([idx])
#     your_idx = np.repeat(your_idx, pillar_feature.shape[0]).reshape((pillar_feature.shape[0],1))
#     pillar_feature = np.concatenate((pillar_feature,your_idx), axis=1)
#     if count:
#         pillar_features = pillar_feature.copy()
#         count = False
#         continue
    
#     pillar_features = np.concatenate((pillar_features,pillar_feature), axis=0)
    
#     if break_time == 100:
#         break

# pillar_feature_3 = pillar_features.copy()
    
    
count = True
pillar_feature_name = "pillar_feature_after_atten/"

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
    
# pillar_features = np.concatenate((pillar_feature_2[:,0:128], pillar_feature_3[:,0:128], pillar_features), axis=1)
      


mat = np.concatenate((pillar_features[:,0:384].astype("float32"), pillar_features[:,385:386].astype("float32")), axis=1)

mat_path = './for_detector_attention' + ".mat"
io.savemat(mat_path, {'name': mat})