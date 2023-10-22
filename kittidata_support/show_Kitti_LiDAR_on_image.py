import view_results as vv
import kitti_util as utils
import numpy as np

idx = 110
image_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/image_2/" + "%06d.png" % idx
# lidar_address_gt = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/velodyne/000001.bin"
# lidar_address = "C:/Users/kk/Desktop/AutonomousCar/PointNet++/Kitti_painted_9class_train/Kitti_val/200epoch_0.3/output_result_nobackground/" + "%06d.bin" % idx
lidar_address = "C:/Users/kk/Desktop/AutonomousCar/PointNet++/Kitti_R+L+LxR/threshold_0.3/output_result_nobackground/" + "%06d.bin" % idx
calib_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/calib/" + "%06d.txt" % idx

pts = utils.load_velo_scan(lidar_address)[:,0:3]

# pts_gt = utils.load_velo_scan(lidar_address_gt)[:,0:3]
img = utils.load_image(image_address)


img0 = img[:,:,0].copy()
img1 = img[:,:,1].copy()
img2 = img[:,:,2].copy()
img[:,:,0] = img2
img[:,:,1] = img1
img[:,:,2] = img0

cal = utils.Calibration(calib_address)
img_height, img_width, img_channel = img.shape

# _,front_pts = vv.show_lidar_on_image(pts_gt,img,cal,img_width,img_height)

vv.show_foreground_lidar_on_image_kitti(pts,img,cal,img_width,img_height)


# aaa = np.zeros((375,1242))
# for i in front_pts:
#     aaa[int(i[1])][int(i[0])] = 1
        