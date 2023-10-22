import view_results as vv
import kitti_util as utils
import cv2

idx = 110

image_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/demo_data3/2011_09_26/2011_09_26_drive_0059_sync/ok/image_2/" + "%06d.png" % idx
pred_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/demo_data3/2011_09_26/2011_09_26_drive_0059_sync/ok/label_2/" + "%06d.txt" % idx
label_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/training/label_2/" + "%06d.txt" % idx
# label_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/kitti_val_level/hard_label_2/" + "%06d.txt" % idx
calib_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/demo_data3/2011_09_26/2011_09_26_drive_0059_sync/ok/calib/" + "%06d.txt" % idx

# image_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/demo_data3/2011_09_26/2011_09_26_drive_0059_sync/ok/image_2/" + "%06d.png" % idx
# label_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/demo_data3/2011_09_26/2011_09_26_drive_0059_sync/ok/label_2/" + "%06d.txt" % idx
# calib_address = "C:/Users/kk/Desktop/AutonomousCar/kittidataset/demo_data3/2011_09_26/2011_09_26_drive_0059_sync/ok/calib/" + "%06d.txt" % idx
lab = utils.read_label(pred_address)
lab_L = utils.read_label(label_address)
img = utils.load_image(image_address)

img0 = img[:,:,0].copy()
img1 = img[:,:,1].copy()
img2 = img[:,:,2].copy()
img[:,:,0] = img2
img[:,:,1] = img1
img[:,:,2] = img0

cal = utils.Calibration(calib_address)

vv.show_image_with_boxes(img,lab,cal)
# vv.show_image_with_boxes_PandL(img,lab,lab_L,cal)

# cv2.imwrite("C:/Users/kk/Desktop/easy_case/" + str(idx) + ".png", cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
