from mytool import tool, kitti
import numpy as np

idx = 13

pts_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/velodyne/" + "%06d.bin" % idx
image_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/image_2/" + "%06d.png" % idx
label_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_2/" + "%06d.txt" % idx
calib_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/calib/" + "%06d.txt" % idx

t = tool()
k = kitti(calib_path)

pts = t.get_lidar(pts_path)[:,0:3]
pts_r = t.get_lidar(pts_path)[:,3:4]
pts_rect = k.get_pts_rect(pts)
img = t.get_img(image_path)
labels = t.get_label(label_path)

pts_rect = pts_rect[k.get_valid_flag(pts,img)]
pts_r = pts_r[k.get_valid_flag(pts,img)]
pts_full = np.concatenate((pts_rect,pts_r),axis=1)

all_object_pts = []
all_object_dis = []
for label in labels:
    valid_label_pts = k.find_valid_label_pts(pts_rect.copy(), label)
    all_object_pts.append(pts_full[valid_label_pts])
    all_object_dis.append(label[2])
    
for i in range(len(all_object_pts)):
    if len(all_object_pts[i]) == 0:continue
    
    object_pts = all_object_pts[i]
    
    # save = object_pts.astype('float32')
    # save.tofile("C:/Users/kk/Desktop/" + str(all_object_dis[i]) + ".bin")