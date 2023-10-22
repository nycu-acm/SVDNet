from mytool import tool, kitti
import kitti_util as utils
import numpy as np
import os

t = tool()
k = kitti()
seg_list = os.listdir("C:/Users/kk/Desktop/full_pred")
counts = len(seg_list)
count = 0
data_root = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/"
save_root = "C:/Users/kk/Desktop/"
for seg in seg_list:
    name = seg.split(".")[0]

    img = t.get_img(data_root + "image_2/" + name + ".png")
    seg = np.load("C:/Users/kk/Desktop/full_pred/" + name + ".npy")
    pts = t.get_lidar(data_root + "velodyne/" + name + ".bin")
    cal = utils.Calibration(data_root + "calib/" + name + ".txt")
    img_shape = (img.shape[1], img.shape[2], 3)
    
    pts_rect = cal.project_velo_to_rect(pts[:,0:3])
    pts_rect_depth = np.squeeze(pts_rect[:,2:])
    pts_img = cal.project_rect_to_image(pts_rect)
    
    
    pts_valid_flag = k.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)
    
    pts_img = pts_img[pts_valid_flag]
    pts_rect = pts_rect[pts_valid_flag]
    pts_original = cal.project_rect_to_velo(pts_rect)
    
    point_seg = np.zeros((pts_img.shape[0],9))
    
    for i in range(pts_img.shape[0]):
        u, v = int(pts_img[i, 0]), int(pts_img[i, 1])
        point_seg[i, 0] = seg[0, v, u]
        point_seg[i, 1] = seg[1, v, u]
        point_seg[i, 2] = seg[2, v, u]
        point_seg[i, 3] = seg[3, v, u]
        point_seg[i, 4] = seg[4, v, u]
        point_seg[i, 5] = seg[5, v, u]
        point_seg[i, 6] = seg[6, v, u]
        point_seg[i, 7] = seg[7, v, u]
        point_seg[i, 8] = seg[8, v, u]
    
    
    save = np.concatenate((pts_original, point_seg), axis=1).astype('float32')
    if not os.path.isdir(save_root + "painted"):
        os.makedirs(save_root + "painted")
    save.tofile(save_root + "painted/" + name + ".bin")
    
    count += 1
    print("Finish: %.2f" % (count/counts*100), "%")
