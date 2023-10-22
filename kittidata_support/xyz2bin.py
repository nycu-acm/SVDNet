import numpy as np

with open( "C:/Users/kk/Desktop/000008.xyz","r") as f:
    pts_list = []
    for line in f.readlines():
        pts_list.append(line)

pts = np.zeros((1,3))
for row in pts_list:
    temp_array = np.zeros((1,3))
    temp_list = row.split(" ")
    temp_array[0][0] = float(temp_list[0])
    temp_array[0][1] = float(temp_list[1])
    temp_array[0][2] = float(temp_list[2])
    pts = np.insert(pts, 0, values=temp_array, axis=0)
    
pts = np.delete(pts,-1,0)

save = pts.astype('float32')
save.tofile("C:/Users/kk/Desktop/000008.bin")
