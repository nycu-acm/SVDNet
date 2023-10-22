import matplotlib.pyplot as plt
from PIL import Image
from mytool import tool, kitti
from tqdm import tqdm
import numpy as np
import os

t = tool()
distance_points_list = []
npoints = 16384

for idx in tqdm(range(len(os.listdir("C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_2/")))):

    pts_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/velodyne/" + "%06d.bin" % idx
    image_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/image_2/" + "%06d.png" % idx
    label_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_2/" + "%06d.txt" % idx
    calib_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/calib/" + "%06d.txt" % idx
    
    k = kitti(calib_path)
    
    img = t.get_img(image_path)
    pts = t.get_lidar(pts_path)[:,0:3]
    pts_rect = k.get_pts_rect(pts)
    pts_rect = pts_rect[k.get_valid_flag(pts,img)]
    
    # if npoints < len(pts_rect):
    #     pts_depth = pts_rect[:, 2]
    #     pts_near_flag = pts_depth < 40.0
    #     far_idxs_choice = np.where(pts_near_flag == 0)[0]
    #     near_idxs = np.where(pts_near_flag == 1)[0]
    #     near_idxs_choice = np.random.choice(near_idxs, npoints - len(far_idxs_choice), replace=False)
    
    #     choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
    #         if len(far_idxs_choice) > 0 else near_idxs_choice
    #     np.random.shuffle(choice)
    # else:
    #     choice = np.arange(0, len(pts_rect), dtype=np.int32)
    #     if npoints > len(pts_rect):
    #         extra_choice = np.random.choice(choice, npoints - len(pts_rect), replace=False)
    #         ##############################################################################True for lyft
    #         choice = np.concatenate((choice, extra_choice), axis=0)
    #     np.random.shuffle(choice)
    
    # pts_rect = pts_rect[choice, :]
    
    labels = t.get_label(label_path, car=True, person=False, van=False)
    
    for label in labels:
        valid_label_pts = k.find_valid_label_pts(pts_rect.copy(), label)
        
        if len(pts_rect[valid_label_pts]) <= 0:
            continue
        
        distance_points_list.append([label[2], len(pts_rect[valid_label_pts])]) # [center distance, point number]
        
distance_points_array = np.asarray(distance_points_list)

# max_distance = max(distance_points_array[:,0])
# max_point_number = max(distance_points_array[:,1])
# distance_points_image = np.zeros((max_point_number+1, max_distance+1))

# for i in distance_points_array:
#     distance_points_image[i[1]][i[0]] = distance_points_image[i[1]][i[0]] + 1    

plt.scatter(distance_points_array[:,0], distance_points_array[:,1])
plt.show()
