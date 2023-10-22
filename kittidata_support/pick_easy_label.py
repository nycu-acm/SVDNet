import os

label_list = os.listdir("C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_2/")

idx = 8
write_txt = []
for i in range(len(label_list)):
    idx = i

    label_path = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_2/" + "%06d.txt" % idx
    save_path = "C:/Users/kk/Desktop/hard_label_2/" + "%06d.txt" % idx
    
    with open(label_path) as f:
        lines = f.readlines()
    
    # write_txt = []
    
    for index in range(len(lines)):
        line = lines[index].split(" ")
        
        class_flag = line[0]=="Car"
        
        # far
        # distance_flag = float(line[13]) >= 40
        
        # # near
        # distance_flag = float(line[13]) < 40
        
        # # all
        # distance_flag = float(line[13]) >= 0
        
        # flag for easy
        # truncated_flag = float(line[1]) <= 0.15
        # occluded_flag = float(line[2]) == 0
        # box_flag = abs(float(line[5]) - float(line[7])) >= 40
        
        # flag for moderate
        # truncated_flag = float(line[1]) <= 0.3
        # occluded_flag = float(line[2]) == 1
        # box_flag = abs(float(line[5]) - float(line[7])) >= 25
        
        # # flag for hard
        # truncated_flag = float(line[1]) <= 0.5
        # occluded_flag = float(line[2]) == 2
        # box_flag = abs(float(line[5]) - float(line[7])) >= 25
        
        
        if class_flag:# & truncated_flag & occluded_flag & box_flag & distance_flag:
            write_txt.append(lines[index])
    
    # with open(save_path, "w") as output:
    #     output.writelines(write_txt)
        
    print("Finish: ", i)