import os
import random
import shutil
from tqdm import tqdm

pts_name = sorted(os.listdir("/data3/waymo_kitti_format/train/velodyne/"))
img_name = sorted(os.listdir("/data3/waymo_kitti_format/train/image_2/"))
label_name = sorted(os.listdir("/data3/waymo_kitti_format/train/label_2/"))
calib_name = sorted(os.listdir("/data3/waymo_kitti_format/train/calib/"))

pts_dest = "/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object4/training/velodyne/"
img_dest = "/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object4/training/image_2/"
label_dest = "/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object4/training/label_2/"
calib_dest = "/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object4/training/calib/"

target_index = []
target_number = 7481


while(len(target_index) < target_number):
    idx = random.choice(label_name).split(".txt")[0]
    
    pts_flag = False
    img_flag = False
    calib_flag = False
    
    if idx + ".bin" in pts_name:
        pts_flag = True
        
    if idx + ".png" in img_name:
        img_flag = True
        
    if idx + ".txt" in calib_name:
        calib_flag = True
        
    if (not pts_flag) or (not img_flag) or (not calib_flag):
        label_name.remove(idx + ".txt")
        continue
    
    target_index.append(idx)
    
    label_name.remove(idx + ".txt")
    
    finish = float(len(target_index) / target_number)*100.0
    
    print("already collect %.2f" % finish, "% data")
    
print("==============  Start copy and rename  ==============")
print('\n')

new_index = 0    
for index in tqdm(target_index):
    
    shutil.copy("/data3/waymo_kitti_format/train/velodyne/" + index + ".bin", pts_dest)
    os.rename(pts_dest + index + ".bin", pts_dest + "%06d.bin" % new_index)
    
    shutil.copy("/data3/waymo_kitti_format/train/image_2/" + index + ".png", img_dest)
    os.rename(img_dest + index + ".png", img_dest + "%06d.png" % new_index)
    
    shutil.copy("/data3/waymo_kitti_format/train/label_2/" + index + ".txt", label_dest)
    os.rename(label_dest + index + ".txt", label_dest + "%06d.txt" % new_index)
    
    shutil.copy("/data3/waymo_kitti_format/train/calib/" + index + ".txt", calib_dest)
    os.rename(calib_dest + index + ".txt", calib_dest + "%06d.txt" % new_index)
    
    new_index = new_index + 1
    