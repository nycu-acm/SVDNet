import os

lyft_file = os.listdir('C:/Users/kk/Desktop/AutonomousCar/DatasetForCompetition/kitti_ver_data/label_2')
#lyft_file = os.listdir('C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_2')
rotation = []

for j in range(1000):
    address_lyft = 'C:/Users/kk/Desktop/AutonomousCar/DatasetForCompetition/kitti_ver_data/label_2/' + lyft_file[j] 
    #address_lyft = 'C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_2/' + lyft_file[j]
    
    with open(address_lyft, 'r') as f:
        lines = f.readlines()
    for i in range(int(len(lines))):
        temp = lines
        temp = temp[i].split(" ")
        if float(temp[12]) == -1000 :
            continue
        rotation.append(float(temp[12]))

range_max = max(rotation)
range_min = min(rotation)