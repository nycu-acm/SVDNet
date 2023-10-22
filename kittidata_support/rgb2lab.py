from skimage import io, color
import numpy as np

for idx in range(7481):
    print("NO.", idx)
# idx = 0 
    filename = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/image_2/" + "%06d.png" % idx
    output_file = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/image_Lab_L/" + "%06d.npy" % idx
    rgb = io.imread(filename)
    lab = color.rgb2lab(rgb)
    
    np.save(output_file, lab[:,:,0])