import view_results as vv
import kitti_util as utils
import numpy as np


image_address = "C:/Users/kk/Desktop/AutonomousCar/DatasetForCompetition/kitti_ver_data/image_2/000017.png"
#lidar_address_gt = "C:/Users/kk/Desktop/AutonomousCar/DatasetForCompetition/kitti_ver_data/velodyne/000834.bin"
#lidar_address = "C:/Users/kk/Desktop/AutonomousCar/PointNet++/Mix_Train/Lyft_val/output_result_nobackground/000834.bin"
lidar_address = "C:/Users/kk/Desktop/AutonomousCar/PointNet++/Lyft_Train/Lyft_val/output_result_nobackground/000834.bin"
calib_address = "C:/Users/kk/Desktop/AutonomousCar/DatasetForCompetition/kitti_ver_data/calib/000017.txt"

pts = utils.load_velo_scan(lidar_address)[:,0:3]
#pts_gt = utils.load_velo_scan(lidar_address_gt)[:,0:3]
img = utils.load_image(image_address)
cal = utils.Calibration(calib_address)
img_height, img_width, img_channel = img.shape

#vv.show_lidar_on_image_lyft(pts_gt,img,cal,img_width,img_height)
vv.show_foreground_lidar_on_image_lyft(pts,img,cal,img_width,img_height)
