from mytool import tool, kitti
import kitti_util as utils
import numpy as np
import csv
import os

with open('./color_mapping.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    
color_mapping = {}    
for i in range(len(data)):
    temp1 = data[i][1].split("(")[1].split(")")[0].split(",")[0]
    temp2 = data[i][1].split("(")[1].split(")")[0].split(",")[1].split(" ")[1]
    temp3 = data[i][1].split("(")[1].split(")")[0].split(",")[2].split(" ")[1]
    key = temp1 + temp2 + temp3   
    value = int(data[i][0])
    

    if not (key in color_mapping):
        color_mapping[key] = value

t = tool()
k = kitti()
img_list = os.listdir("C:/Users/kk/Desktop/segmentation_result")
counts = len(img_list)
count = 0

for img in img_list:
    name = img.split(".")[0]

    seg = t.get_img("C:/Users/kk/Desktop/segmentation_result/" + name +".png")
    seg = seg[:,:,int(seg.shape[2]/2):]
    pts = t.get_lidar("C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/velodyne/" + name + ".bin")
    cal = utils.Calibration("C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/calib/" + name + ".txt")
    img_shape = (seg.shape[1], seg.shape[2], 3)
    
    pts_rect = cal.project_velo_to_rect(pts[:,0:3])
    pts_intensity = pts[:, 3]
    pts_rect_depth = np.squeeze(pts_rect[:,2:])
    pts_img = cal.project_rect_to_image(pts_rect)
    
    
    pts_valid_flag = k.get_valid_flag(pts_rect, pts_img, pts_rect_depth, img_shape)
    
    pts_img = pts_img[pts_valid_flag]
    pts_rect = pts_rect[pts_valid_flag]
    pts_original = cal.project_rect_to_velo(pts_rect)
    
    point_seg = np.zeros((pts_img.shape[0],3))
    seg_class = np.zeros((pts_img.shape[0],1))
    
    for i in range(pts_img.shape[0]):
        u, v = int(pts_img[i, 0]), int(pts_img[i, 1])
        point_seg[i, 0] = seg[0, v, u]
        point_seg[i, 1] = seg[1, v, u]
        point_seg[i, 2] = seg[2, v, u]
    
    for i in range(pts_img.shape[0]):
        index = str(int(point_seg[i, 0]))+str(int(point_seg[i, 1]))+str(int(point_seg[i, 2]))
        seg_class[i, 0] = color_mapping[index]
    
    save = np.concatenate((pts_original, seg_class), axis=1).astype('float32')
    save.tofile("C:/Users/kk/Desktop/painted/" + name + ".bin")
    
    count += 1
    print("Finish: %.2f" % (count/counts*100), "%")