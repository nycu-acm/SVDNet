import numpy as np
import kitti_util as utils


address_kitti = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/velodyne/000017.bin"
#calib_dir = "C:/Users/kk/Desktop/AutonomousCar/DatasetForCompetition/kitti_ver_data/velodyne/000017.bin"
address_kitti2 = "C:/Users/kk/Desktop/AutonomousCar/DatasetForCompetition/kitti_ver_data/velodyne/000017.bin"

pts_kitti_check = np.fromfile(address_kitti, dtype=np.float32).reshape(-1, 4)
pts_kitti_check2 = np.fromfile(address_kitti2, dtype=np.float32).reshape(-1, 4)

# calib = utils.Calibration(calib_dir)
# pts_rect = calib.project_velo_to_rect(pts_kitti_check2[:,0:3])