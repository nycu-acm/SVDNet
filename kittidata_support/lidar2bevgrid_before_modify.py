from mytool import tool, kitti
import numpy as np
import cv2
# 7481個點雲最大值
# bev_depth_size = 69.12 #1.35~80 for car
# bev_width_size = 79.36 #45~-45 for car
bev_depth_size = 47.36 #1.35~80 for person
bev_width_size = 39.68 #45~-45 for person

#正方形
grid_size = 0.16

#pillar個數
bev_depth_grid_num = int(np.ceil(bev_depth_size/grid_size))
bev_width_grid_num = int(np.ceil(bev_width_size/grid_size))

#BEV pillar 原型
# bev_grid = []
# single_row_grid = []

# for i in range(bev_width_grid_num):
#     single_row_grid.append(np.zeros((1,3)).copy())

# for i in range(bev_depth_grid_num):
#     bev_grid.append(single_row_grid.copy())

#把點雲 pillar化
# idx = 8
for idx in range(7481):
    print("now process NO.", "%06d.png" % idx)
    pts_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/velodyne/" + "%06d.bin" % idx
    image_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/image_2/" + "%06d.png" % idx
    label_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_2/" + "%06d.txt" % idx
    calib_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/calib/" + "%06d.txt" % idx
    
    t = tool()
    k = kitti(calib_path)
    
    pts = t.get_lidar(pts_path)[:,0:3]
    pts_rect = k.get_pts_rect(pts)
    img = t.get_img(image_path)
    labels = t.get_label(label_path)
    
    pts_rect = pts_rect[k.get_valid_flag(pts,img)]
    
    # grid_depth_index = 0
    # for depth in range(bev_depth_grid_num):
    #     pts_lower_bound = pts_rect[:,2] >= depth*grid_size
    #     pts_upper_bound = pts_rect[:,2] < (depth + 1)*grid_size
    #     pts_current_depth = pts_rect[pts_lower_bound & pts_upper_bound]
        
    #     if pts_current_depth.shape[0]==0:
    #         grid_depth_index += 1
    #     else:
    #         grid_width_index = 0
    #         for width in range(bev_width_grid_num):
    #             pts_left_bound = pts_current_depth[:,0] >= (-(bev_width_size/2) + width*grid_size)
    #             pts_right_bound = pts_current_depth[:,0] < (-(bev_width_size/2) + (width + 1)*grid_size)
    #             pts_current_width = pts_current_depth[pts_left_bound & pts_right_bound]
                
    #             if pts_current_width.shape[0]==0:
    #                 grid_width_index += 1
    #             else:
    #                 bev_grid[bev_depth_grid_num - 1 - grid_depth_index][grid_width_index] = pts_current_width
    #                 grid_width_index += 1
                    
    #         grid_depth_index += 1
    
    
    #BEV pillar label 原型
    # bev_label_grid = []
    # single_row_label_grid = []
    
    # for i in range(bev_width_grid_num):
    #     single_row_label_grid.append(0)
    
    # for i in range(bev_depth_grid_num):
    #     bev_label_grid.append(single_row_label_grid.copy())
    
    
    # #用BBOX中心點去label grid
    # center_label_grid = np.zeros((bev_depth_grid_num,bev_width_grid_num))
    # # X:width Y:height Z:depth
    # for label in labels:
    #     bbox_x = label[0]
    #     bbox_z = label[2]
        
    #     if ((bbox_x > 45) or (bbox_x < -45) or (bbox_z > 80) or (bbox_z < 1.35)):
    #         pass
    #     else:
    #         bbox_width_grid_index = int((bbox_x + 45)// grid_size)
    #         bbox_depth_grid_index = int(bbox_z // grid_size)
    #         center_label_grid[bev_depth_grid_num - 1 - bbox_depth_grid_index][bbox_width_grid_index] = 1
    #         bev_label_grid[bev_depth_grid_num - 1 - bbox_depth_grid_index][bbox_width_grid_index] = 1
            
    pts_grid = np.zeros((bev_depth_grid_num,bev_width_grid_num))
    
    for pts in pts_rect:
    
        pts_x = pts[0]
        pts_z = pts[2]
        
        # for car
        # if ((pts_x > 39.68) or (pts_x < -39.68) or (pts_z > 69.12) or (pts_z < 0)):
        #     pass
        
        # for person
        if ((pts_x > 19.84) or (pts_x < -19.84) or (pts_z > 47.36) or (pts_z < 0)):
            pass
        else:
            # for car
            # pts_width_grid_index = int((pts_x + 39.68)// grid_size)
            
            # for person
            pts_width_grid_index = int((pts_x + 19.84)// grid_size)
            pts_depth_grid_index = int(pts_z // grid_size)
            
            # 有點的grid
            pts_grid[bev_depth_grid_num - 1 - pts_depth_grid_index][pts_width_grid_index] = 1
                
    
    
    #用物件的所有點去label grid
    pts_label_grid = np.ones((bev_depth_grid_num,bev_width_grid_num))
    pts_label_grid = pts_grid - pts_label_grid.copy()
    # 得到物件的點
    all_object_pts = []
    for label in labels:
        valid_label_pts = k.find_valid_label_pts(pts_rect.copy(), label)
        all_object_pts.append(pts_rect[valid_label_pts])
        
    for object_pts in all_object_pts:
        for single_pts in object_pts:
            pts_x = single_pts[0]
            pts_z = single_pts[2]
            
            # for car
            # if ((pts_x > 39.68) or (pts_x < -39.68) or (pts_z > 69.12) or (pts_z < 0)):
            #     pass
            
            # for person
            if ((pts_x > 19.84) or (pts_x < -19.84) or (pts_z > 47.36) or (pts_z < 0)):
                pass
            else:
                # for car
                # pts_width_grid_index = int((pts_x + 39.68)// grid_size)
                
                # for person
                pts_width_grid_index = int((pts_x + 19.84)// grid_size)
                pts_depth_grid_index = int(pts_z // grid_size)
                
                pts_label_grid[bev_depth_grid_num - 1 - pts_depth_grid_index][pts_width_grid_index] += 1
                # bev_label_grid[bev_depth_grid_num - 1 - pts_depth_grid_index][pts_width_grid_index] = 1
                
    np.save("C:/Users/kk/Desktop/label_grid_num_person/"  + "%06d.npy" % idx, pts_label_grid[:,:,np.newaxis])
    # cv2.imwrite("C:/Users/kk/Desktop/label_grid/"  + "%06d.png" % idx, pts_label_grid*255 )



 