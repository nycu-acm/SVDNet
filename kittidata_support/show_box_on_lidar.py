import view_results as vv
import kitti_util as utils
import numpy as np

idx = 4

# label_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/demo_data3/2011_09_26/2011_09_26_drive_0059_sync/ok/label_2_16bin/" + "%06d.txt" % idx
# # pred_address = "C:/Users/kk/Desktop/PointPillars_person/" + "%06d.txt" % idx
# # pred_address = "C:/Users/kk/Desktop/D-PointPillars_person/" + "%06d.txt" % idx
# # pred_address = "C:/Users/kk/Desktop/step_441728/" + "%06d.txt" % idx
# calib_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/demo_data3/2011_09_26/2011_09_26_drive_0059_sync/ok/calib/" + "%06d.txt" % idx
# lidar_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/demo_data3/2011_09_26/2011_09_26_drive_0059_sync/velodyne_points/data/"  + "%06d.bin" % idx

# pred_address = "C:/Users/kk/Desktop/step_441728/" + "%06d.txt" % idx
# label_address = "C:/Users/kk/Desktop/label_2/" + "%06d.txt" % idx
# calib_address = "C:/Users/kk/Desktop/calib/" + "%06d.txt" % idx
# lidar_address = "C:/Users/kk/Desktop/velodyne_reduced/" + "%06d.bin" % idx

# pred_address = "C:/Users/kk/Desktop/D-PointPillars/" + "%06d.txt" % idx
# pred_address = "C:/Users/kk/Desktop/attention/" + "%06d.txt" % idx
# pred_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_2/" + "%06d.txt" % idx
calib_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/calib/" + "%06d.txt" % idx
# lidar_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/velodyne/"  + "%06d.bin" % idx
label_address = "C:/Users/kk/Desktop/" + "%06d.txt" % idx

# pts = utils.load_velo_scan(lidar_address)

pts = np.random.rand(100,4)
cal = utils.Calibration(calib_address)

pts_flag1 = pts[:,0]>=0
pts_flag2 = pts[:,0]<=69.12
pts_flag3 = pts[:,1]>=-39.68
pts_flag4 = pts[:,1]<=39.68
pts_flag5 = pts[:,2]>=-16
pts_flag6 = pts[:,2]<=16
pts_flag = pts_flag1 & pts_flag2 & pts_flag3 & pts_flag4 & pts_flag5 & pts_flag6 
# pts = cal.project_rect_to_velo(pts_3d_rect = pts[:,0:3])

# lab = utils.read_label(pred_address)
lab_L = utils.read_label(label_address)

vv.show_lidar_with_boxes(pts[pts_flag],lab_L,cal)
# vv.show_lidar_with_boxes_PandL(pts[pts_flag],lab,lab_L,cal)