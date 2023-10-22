import numpy as np
import cv2
import glob
import os


def image2mp4 (input_address,input_address_2,fps,image_length,image_width):
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    videoWriter = cv2.VideoWriter('vedio.mp4',fourcc,fps,(image_length,image_width))
    # imgs=sorted(glob.glob(input_address + '\\*.png'))
    for imgname in range(1, (len(os.listdir(input_address)) + 1)):
        frame1 = cv2.imread(input_address + str(imgname) + ".png")
        # frame2 = cv2.imread(input_address_2 + str(imgname) + ".png")
        # frame = np.concatenate((frame1,frame2), axis=1)
        videoWriter.write(frame1)
    
    videoWriter.release()
    cv2.destroyAllWindows()
    
# 改根目錄
address = 'D:\\scene_01_pred\\combine\\'
address2 = 'D:\\Allring_dataset_625\\scene_03\\depth\\'

fps = 8.0
image_length = 320
image_width = 240

image2mp4(address,address2,fps,image_length,image_width)
cv2.destroyAllWindows()