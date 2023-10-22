from mytool import tool, kitti
import numpy as np

idx = 13

pts_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/velodyne/" + "%06d.bin" % idx
label_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_2/" + "%06d.txt" % idx
calib_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/calib/" + "%06d.txt" % idx

t = tool()
k = kitti(calib_path)



pts = t.get_lidar(pts_path)[:,0:3]
pts_r = t.get_lidar(pts_path)[:,3:4]
pts_rect = k.get_pts_rect(pts)
labels = t.get_label(label_path)

# pts_flag1 = pts[:,2]>=0
# pts_flag2 = pts[:,2]<=69.12
# pts_flag3 = pts[:,0]>=-39.68
# pts_flag4 = pts[:,0]<=39.68
# pts_flag5 = pts[:,1]>=-3
# pts_flag6 = pts[:,1]<=1
# pts_flag = pts_flag1 & pts_flag2 & pts_flag3 & pts_flag4 & pts_flag5 & pts_flag6 

# pts_rect = pts_rect[pts_flag]
# pts_r = pts_r[pts_flag]
pts_full = np.concatenate((pts_rect,pts_r),axis=1)

all_object_pts = []
all_object_dis = []

for label in labels:
    valid_label_pts = k.find_valid_label_pts(pts_rect.copy(), label)
    all_object_pts.append(pts_rect[valid_label_pts])
    all_object_dis.append(label[2])
    
# for i in range(len(all_object_pts)):
#     if len(all_object_pts[i]) == 0:continue
    
#     object_pts.append(all_object_pts[i])
    