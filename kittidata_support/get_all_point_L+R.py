from mytool import tool, kitti
import kitti_util as utils
import numpy as np
import os

for idx in range(7481):
    print("NO.", idx)
# idx = 7
    pts_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/velodyne/" + "%06d.bin" % idx
    image_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/image_2/" + "%06d.png" % idx
    calib_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/calib/" + "%06d.txt" % idx
    img_l_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/image_Lab_L/" + "%06d.npy" % idx
    save_path = "C:/Users/kk/Desktop/L+R_train/" + "%06d.npy" % idx
    save_root = "C:/Users/kk/Desktop/"
    
    t = tool()
    k = kitti(calib_path)
    
    img_l = np.load(img_l_path)
    img = t.get_img(image_path)
    pts = t.get_lidar(pts_path)
    cal = utils.Calibration(calib_path)
    
    pts_rect = cal.project_velo_to_rect(pts[:,0:3])
    pts_img = cal.project_rect_to_image(pts_rect)

    pts_valid_flag = k.get_valid_flag(pts[:,0:3], img)
    
    pts_img = pts_img[pts_valid_flag]
    pts = pts[pts_valid_flag]
    
    point_l = np.zeros((pts_img.shape[0],1))
    
    for i in range(pts_img.shape[0]):
        u, v = int(pts_img[i, 0]), int(pts_img[i, 1])
        point_l[i, 0] = img_l[v, u]
    
    save = np.concatenate((pts, point_l), axis=1)
    
    if not os.path.isdir(save_root + "L+R_train"):
        os.makedirs(save_root + "L+R_train")
    np.save(save_path, save)