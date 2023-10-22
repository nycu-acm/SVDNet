import os
import numpy as np
import kitti_util 

class tool():
    def __init__(self,):
        pass
    
    def get_img(self, path):
        from PIL import Image      
        return np.transpose(Image.open(path), (2, 0, 1))
    
    def get_lidar(self, path):
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    
    def get_label(self, path, car=True, person=False, van=False):
        
        # 讀kitti的label資料
        with open(path, 'r') as f:
            lines = f.readlines()
        objects = [kitti_util.Object3d(line) for line in lines]
        
        type_whitelist = []
        if person:
            type_whitelist.append("Pedestrian")
            type_whitelist.append("Cyclist")
        if car:
            type_whitelist.append("Car")
        if van:
            type_whitelist.append("Van")
        valid_obj_list = []
        for obj in objects:
            if obj.type not in type_whitelist:
                continue
            valid_obj_list.append(obj)
        
       # 得到 x y z h w l angle score
        boxes3d = np.zeros((valid_obj_list.__len__(), 8), dtype=np.float32)
        for k, obj in enumerate(valid_obj_list):
            boxes3d[k, 0:3], boxes3d[k, 3], boxes3d[k, 4], boxes3d[k, 5], boxes3d[k, 6], boxes3d[k, 7] \
                = obj.t, obj.h, obj.w, obj.l, obj.ry, obj.score
        return boxes3d
    
    def png2jpg(self, img_file_path):
        img_list = os.listdir(img_file_path)
        img_list_after = []
        
        for i in range(len(img_list)):  
            img_list_after.append(img_list[i].split(".")[0] + ".jpg")
        
        for i in range(len(img_list)):               
            now = img_file_path + "/" + img_list[i]
            after = img_file_path + "/" + img_list_after[i]
            
            if os.path.isfile(now):
                os.rename(now,after)

            print("Finish: %.2f" % ((i + 1)/len(img_list)*100), "%")
            
    def pts_rotate(self, pts, r_angle):
        r_matrix = np.array([[np.cos(r_angle), 0, np.sin(r_angle)],
                      [0, 1, 0],
                      [-np.sin(r_angle), 0, np.cos(r_angle)]])
        
        pts_after = np.dot(r_matrix, pts.T).T
        return pts_after
        
    
class kitti():
    def __init__(self,calib_path):
        self.calib = kitti_util.Calibration(calib_path)
    
    def get_pts_rect(self, pts):
        pts_rect = self.calib.project_velo_to_rect(pts)
        return pts_rect
    
    def get_depth_map(self, pts):
        pts_rect = self.get_pts_rect(pts)
        depth_map = self.calib.project_rect_to_image(pts_rect)
        return depth_map
    
    def get_valid_flag(self, original_pts, img):
        pts_rect = self.get_pts_rect(original_pts)
        depth_map = self.get_depth_map(original_pts)
        pts_rect_depth = np.squeeze(pts_rect[:,2:])
        img_shape = (img.shape[1], img.shape[2], 3)
        val_flag_1 = np.logical_and(depth_map[:, 0] >= 0, depth_map[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(depth_map[:, 1] >= 0, depth_map[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
        
        # kitti lidar (camera view)      1.358 < depth < 79.718  -51.734 < width < 52.54
        # kitti label (camera view)      -0.57 < depth < 103.6    -44.03 < width < 40.06
        # x_range, y_range, z_range = [[-39.68, 39.68], [-3,   1], [0, 69.12]]
        
        
        x_range, y_range, z_range = [[-51.734, 52.54], [-5,   5], [0, 100]]
        pts_x, pts_y, pts_z = pts_rect[:, 0], pts_rect[:, 1], pts_rect[:, 2]
        range_flag = (pts_x >= x_range[0]) & (pts_x <= x_range[1]) \
                      & (pts_y >= y_range[0]) & (pts_y <= y_range[1]) \
                      & (pts_z >= z_range[0]) & (pts_z <= z_range[1])
        pts_valid_flag = range_flag & pts_valid_flag
        
        return pts_valid_flag
       
    def find_valid_label_pts(self, pts_rect, single_label):         # single label = [x y z h w l angle]
        corner3d = self.generate_corners3d(single_label)
        valid_label_pts = self.in_hull(pts_rect, corner3d)

        return valid_label_pts
    
    def generate_corners3d(self, sigle_label):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = sigle_label[5], sigle_label[3], sigle_label[4]
        ry = sigle_label[6]
        pos = sigle_label[0:3]
        
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(ry), 0, np.sin(ry)],
                      [0, 1, 0],
                      [-np.sin(ry), 0, np.cos(ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + pos
        
        # only select 3d bounding box for objs in front of the camera
        if np.any(corners3d[2,:]<0.1):
            return(np.zeros((8,3))-1)
        else:
            return corners3d    
    
    def in_hull(self,p , hull):
        from scipy.spatial import Delaunay
        import scipy
        """
        :param p: (N, 3D) pts points
        :param hull: (M, 3D) M corners of a box (usually M = 8)
        :return (N) bool
        """
        try:
            if not isinstance(hull, Delaunay):
                hull = Delaunay(hull)
            flag = hull.find_simplex(p) >= 0
        except scipy.spatial.qhull.QhullError:
            # print('Warning: not a hull %s' % str(hull))
            flag = np.zeros(p.shape[0], dtype=np.bool)
    
        return flag