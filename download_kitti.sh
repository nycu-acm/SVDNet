#!/bin/bash
echo "download start"

wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip

echo "download end"

echo "unzip start"
unzip data_object_calib.zip
unzip data_object_image_2.zip
unzip data_object_label_2.zip
unzip data_object_velodyne.zip
mkdir training/velodyne_reduced
mkdir testing/velodyne_reduced

mkdir KITTI_DATASET_ROOT
mv training KITTI_DATASET_ROOT
mv testing KITTI_DATASET_ROOT

echo "unzip end"

echo "ALL ARE COMPLETE
