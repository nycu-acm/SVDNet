import kitti_util as utils

count = 0

# Easy
min_height = 40.0
max_occlusion = 0.0
max_truncation = 0.15

# Mod.
# min_height = 25.0
# max_occlusion = 1.0
# max_truncation = 0.30

# Hard
# min_height = 25.0
# max_occlusion = 2.0
# max_truncation = 0.50



distance_threshold = 45.0


for idx in range(7481):
    label_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_2/" + "%06d.txt" % idx
    objects = utils.read_label(label_address)
    for obj in objects:
        if obj.type=='DontCare':continue
        if obj.type!='Car':continue
        
        if abs(obj.ymax - obj.ymin) < min_height:continue
        if obj.occlusion > max_occlusion:continue
        if obj.truncation > max_truncation:continue
        if obj.t[2] < distance_threshold:continue
        print(idx)
        count += 1
        
print("target number: ", count)

