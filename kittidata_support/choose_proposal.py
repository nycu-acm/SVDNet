label_address = "C:/Users/kk/Desktop/000050.txt"

with open(label_address, 'r') as f:
    lines = f.readlines()
write_index = []
mean_angle = 0.0
count = 1
for i in range(int(len(lines))):
    temp = lines
    temp = temp[i].split(" ")
    
    if float(temp[13]) == 33.69:
        mean_angle = mean_angle*count + float(temp[14])       
        if count == 1:
            write_index.append(i)
        
        count = count + 1
        mean_angle = mean_angle / count
        
        if ((float(temp[14]) - mean_angle) > 0.09) or ((float(temp[14]) - mean_angle) < -0.09):
            write_index.append(i)
    
    print("Finish NO.",i)
    
with open(label_address, 'w') as f:
    for j in range(int(len(write_index))):
        f.write(lines[write_index[j]])

