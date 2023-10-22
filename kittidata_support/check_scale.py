import numpy as np
import os
from tqdm import tqdm

pts_kitti_depth_max = []
pts_kitti_depth_min = []
pts_kitti_width_max = []
pts_kitti_width_min = []
pts_kitti_height_max = []
pts_kitti_height_min = []
# pts_lyft_depth_max = []
# pts_lyft_depth_min = []
# pts_lyft_width_max = []
# pts_lyft_width_min = []
# pts_lyft_height_max = []
# pts_lyft_height_min = []

kitti_file = os.listdir('C:/Users/kk/Desktop/velodyne/')
# lyft_file = os.listdir('C:/Users/kk/Desktop/AutonomousCar/PointNet++/Lyft_Train/Lyft_val/output_result')

for i in tqdm(range(1000)):
    
    x = i
    
    address_kitti = 'C:/Users/kk/Desktop/velodyne/' + kitti_file[x]
    # address_lyft = 'C:/Users/kk/Desktop/AutonomousCar/PointNet++/Lyft_Train/Lyft_val/output_result/' + lyft_file[x]
    
    
    pts_kitti = np.fromfile(address_kitti, dtype=np.float32).reshape(-1, 4)
    pts_kitti_depth = pts_kitti[:,0]
    pts_kitti_width = pts_kitti[:,1]
    pts_kitti_height = pts_kitti[:,2]

    pts_kitti_depth_max.append(max(pts_kitti_depth))
    pts_kitti_depth_min.append(min(pts_kitti_depth))
    pts_kitti_width_max.append(max(pts_kitti_width))
    pts_kitti_width_min.append(min(pts_kitti_width))
    pts_kitti_height_max.append(max(pts_kitti_height))
    pts_kitti_height_min.append(min(pts_kitti_height))

    # pts_lyft = np.fromfile(address_lyft, dtype=np.float32).reshape(-1, 4)
    # pts_lyft_depth = pts_lyft[:,2]
    # pts_lyft_width = pts_lyft[:,0]
    # pts_lyft_height = pts_lyft[:,1]
    # pts_lyft_depth_max.append(max(pts_lyft_depth))
    # pts_lyft_depth_min.append(min(pts_lyft_depth))
    # pts_lyft_width_max.append(max(pts_lyft_width))
    # pts_lyft_width_min.append(min(pts_lyft_width))
    # pts_lyft_height_max.append(max(pts_lyft_height))
    # pts_lyft_height_min.append(min(pts_lyft_height))
    
    
pts_kitti_depth_max_mean = sum(pts_kitti_depth_max) / len(pts_kitti_depth_max)
pts_kitti_depth_min_mean = sum(pts_kitti_depth_min) / len(pts_kitti_depth_min)
pts_kitti_width_max_mean = sum(pts_kitti_width_max) / len(pts_kitti_width_max)
pts_kitti_width_min_mean = sum(pts_kitti_width_min) / len(pts_kitti_width_min)
pts_kitti_height_max_mean = sum(pts_kitti_height_max) / len(pts_kitti_height_max)
pts_kitti_height_min_mean = sum(pts_kitti_height_min) / len(pts_kitti_height_min)
# pts_lyft_depth_max_mean = sum(pts_lyft_depth_max) / len(pts_lyft_depth_max)
# pts_lyft_depth_min_mean = sum(pts_lyft_depth_min) / len(pts_lyft_depth_min)
# pts_lyft_width_max_mean = sum(pts_lyft_width_max) / len(pts_lyft_width_max)
# pts_lyft_width_min_mean = sum(pts_lyft_width_min) / len(pts_lyft_width_min)
# pts_lyft_height_max_mean = sum(pts_lyft_height_max) / len(pts_lyft_height_max)
# pts_lyft_height_min_mean = sum(pts_lyft_height_min) / len(pts_lyft_height_min)