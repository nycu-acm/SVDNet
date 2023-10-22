from mytool import tool,kitti
import kitti_util as utils
import numpy as np

t = tool()
k = kitti()
img = t.get_img("C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/image_2/000008.png")
pts = t.get_lidar("C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/velodyne/000008.bin")
cal = utils.Calibration("C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/calib/000008.txt")
img_shape = (img.shape[1], img.shape[2], 3)

pts_rect = cal.project_velo_to_rect(pts[:,0:3])
pts_rect_depth = np.squeeze(pts_rect[:,2:])
pts_img = cal.project_rect_to_image(pts_rect)


pts_valid_flag = k.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)

pts_img = pts_img[pts_valid_flag]
pts_rect = pts_rect[pts_valid_flag]
pts_original = cal.project_rect_to_velo(pts_rect)

pts_list = []
for row in pts_original:
    pts_list.append(str(row[0]) + " " + str(row[1]) + " " + str(row[2]))

f = open("C:/Users/kk/Desktop/000008.txt", "w")
for row in pts_list:
    f.writelines(row + "\n")
f.close()

