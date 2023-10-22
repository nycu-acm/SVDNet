from mytool import tool, kitti
import kitti_util as utils
import numpy as np
import cv2

for idx in range(7481):
    print("NO.", idx)
# idx = 7
    pts_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/velodyne/" + "%06d.bin" % idx
    image_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/image_2/" + "%06d.png" % idx
    calib_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/calib/" + "%06d.txt" % idx
    img_l_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/image_Lab_L/" + "%06d.npy" % idx
    save_path = "C:/Users/kk/Desktop/energy_map/" + "%06d.png" % idx

    
    t = tool()
    k = kitti(calib_path)
    
    img_l = np.load(img_l_path)
    
    energy_map = np.zeros(img_l.shape)
    output = np.zeros((img_l.shape[0], img_l.shape[1], 3))
    
    img = t.get_img(image_path) 
    pts = t.get_lidar(pts_path)
    cal = utils.Calibration(calib_path)
    
    pts_rect = cal.project_velo_to_rect(pts[:,0:3])
    pts_img = cal.project_rect_to_image(pts_rect)

    pts_valid_flag = k.get_valid_flag(pts[:,0:3], img)
    
    pts_img = pts_img[pts_valid_flag]
    pts = pts[pts_valid_flag]
    
    for i in range(pts_img.shape[0]):
        u, v = int(pts_img[i, 0]), int(pts_img[i, 1])
        energy_map[v, u] = img_l[v, u]*pts[i, 3]
        if energy_map[v, u] != 0:
            energy_map[v, u] = energy_map[v, u] * 2.56
        
    output[:,:,0] = energy_map.copy()
    output[:,:,1] = energy_map.copy()
    output[:,:,2] = energy_map.copy()
    
    cv2.imwrite(save_path, output)