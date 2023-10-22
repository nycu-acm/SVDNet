from mytool import tool, kitti
import numpy as np
import os

for idx in range(7481):
    print("NO.", idx)
# idx = 7
    pts_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/velodyne/" + "%06d.bin" % idx
    image_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/image_2/" + "%06d.png" % idx
    label_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_2/" + "%06d.txt" % idx
    calib_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/calib/" + "%06d.txt" % idx
    save_path = "C:/Users/kk/Desktop/L+R_gt/" + "%06d.npy" % idx
    save_root = "C:/Users/kk/Desktop/"
    
    
    t = tool()
    k = kitti(calib_path)
    
    pts = t.get_lidar(pts_path)
    pts_xyz = pts[:,0:3].copy()
    pts_r = pts[:,3:4].copy()
    pts_xyz_rect = k.get_pts_rect(pts_xyz)
    img = t.get_img(image_path)
    labels = t.get_label(label_path)
    pts_xyz_rect = np.concatenate((pts_xyz_rect,pts_r), axis=1)
    pts_xyz_rect = pts_xyz_rect[k.get_valid_flag(pts_xyz,img)]
    
    if len(labels)==0:
        ground_truth = np.zeros((pts_xyz_rect.shape[0],1))
    
    hint = True
    for label in labels:
        valid_label_pts = k.find_valid_label_pts(pts_xyz_rect[:,0:3].copy(), label)
        if hint:
            ground_truth = valid_label_pts.copy().astype(int)
            hint = False
        else:
            ground_truth = ground_truth + valid_label_pts.copy().astype(int)
            
    np.save(save_path, ground_truth)
            
    # ground_truth = ground_truth[:,np.newaxis].astype('float32')
    
    # if not os.path.isdir(save_root + "L+R_gt"):
    #     os.makedirs(save_root + "L+R_gt")
    # ground_truth.tofile(save_path)
