# SVDNet
## Webpage
http://acm.cs.nctu.edu.tw/Demo/data/013_SVDNet/SVDNet.html

## Note
1. Init this repository from 2021 graduated CHEN CHIH JEN project.
2. In the OpenPCDet_ros repository(2022 graduated CHANG MING JEN), there is related code. (./pcdet/models/necks)

## 使用文件
1. 開環境
2. cd /data2/chihjen/second.pytorch/second/
3. export PYTHONPATH=$PYTHONPATH:/data2/chihjen/second.pytorch/
4. CUDA_VISIBLE_DEVICES=0 pytorch/train.py train --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/path/to/model_dir
5. CUDA_VISIBLE_DEVICES=0 pytorch/train.py evaluate --config_path=./configs/pointpillars/car/xyres_16.proto --model_dir=/path/to/model_dir

> 註:
> 1. model_dir 後面放完整路徑到目標資料夾 (weight放裡面sssss)
> 2. 要painting 要改code 要改那些參考 create_data_setting.txt
> 3. 改model 就改 "/data2/chihjen/second.pytorch/second/pytorch/models/voxelnet.py"
> 4. 要eval 不同距離  就打開 "/data2/chihjen/second.pytorch/second/utils/eval.py" 797~850 
> 5. pillar.yaml 環境備份

## Create data setting
1. ./data/kitti_common.py   =>  line.113  
2. ./create_data.py         =>  line.28  line.231  line.144
3. create  velodyne_reduced  file

## Files about SVDnet (directly)
1. .\second.pytorch\second\pytorch\train.py
2. .\second.pytorch\second\pytorch\builder\second_builder.py
3. .\second.pytorch\second\pytorch\builder\second_builder.py
4. .\second.pytorch\second\pytorch\models\voxelnet.py
