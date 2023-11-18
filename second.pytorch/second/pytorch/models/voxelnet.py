import time
from enum import Enum
from functools import reduce
import numpy as np
import sparseconvnet as scn
import torch
from torch import nn
from torch.nn import functional as F
import math
import torchplus
from torchplus import metrics
from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.ops.array_ops import gather_nd, scatter_nd
from torchplus.tools import change_default_args
from second.pytorch.core import box_torch_ops
from second.pytorch.core.losses import (WeightedSigmoidClassificationLoss,
                                          WeightedSmoothL1LocalizationLoss,
                                          WeightedSoftmaxClassificationLoss)
from second.pytorch.models.pointpillars import PillarFeatureNet, PointPillarsScatter
from second.pytorch.models.foreground_grid_detect import SigmoidFocalClassificationLoss
# from second.pytorch.models.distance2ratio import distance2ratio, only_distance2ratio
from second.pytorch.utils import get_paddings_indicator
from second.pytorch.models.dyrelu import DyReLUB
import random


def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class]
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(
            batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, name='vfe'):
        super(VFELayer, self).__init__()
        self.name = name
        self.units = int(out_channels / 2)
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        pointwise = F.relu(x)
        # [K, T, units]

        aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        # [K, 1, units]
        repeated = aggregated.repeat(1, voxel_count, 1)

        concatenated = torch.cat([pointwise, repeated], dim=2)
        # [K, T, 2 * units]
        return concatenated


class VoxelFeatureExtractor(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        assert len(num_filters) == 2
        num_input_features += 3  # add mean features
        
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        self.vfe1 = VFELayer(num_input_features, num_filters[0], use_norm)
        self.vfe2 = VFELayer(num_filters[0], num_filters[1], use_norm)
        self.linear = Linear(num_filters[1], num_filters[1])
        # var_torch_init(self.linear.weight)
        # var_torch_init(self.linear.bias)
        self.norm = BatchNorm1d(num_filters[1])

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat(
                [features, features_relative, points_dist], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        # mask = features.max(dim=2, keepdim=True)[0] != 0
        
        x = self.vfe1(features) 
              
        x *= mask
        x = self.vfe2(x)
        x *= mask
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()
        x = F.relu(x)
        x *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = torch.max(x, dim=1)[0]
        return voxelwise


class VoxelFeatureExtractorV2(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=[32, 128],
                 with_distance=False,
                 name='VoxelFeatureExtractor'):
        super(VoxelFeatureExtractorV2, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        assert len(num_filters) > 0
        num_input_features += 3
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        num_filters = [num_input_features] + num_filters
        filters_pairs = [[num_filters[i], num_filters[i + 1]]
                         for i in range(len(num_filters) - 1)]
        self.vfe_layers = nn.ModuleList(
            [VFELayer(i, o, use_norm) for i, o in filters_pairs])
        self.linear = Linear(num_filters[-1], num_filters[-1])
        # var_torch_init(self.linear.weight)
        # var_torch_init(self.linear.bias)
        self.norm = BatchNorm1d(num_filters[-1])

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat(
                [features, features_relative, points_dist], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        for vfe in self.vfe_layers:
            features = vfe(features)
            features *= mask
        features = self.linear(features)
        features = self.norm(features.permute(0, 2, 1).contiguous()).permute(
            0, 2, 1).contiguous()
        features = F.relu(features)
        features *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = torch.max(features, dim=1)[0]
        return voxelwise


class SparseMiddleExtractor(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SparseMiddleExtractor'):
        super(SparseMiddleExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        sparse_shape = np.array(output_shape[1:4]) + [1, 0, 0]
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.scn_input = scn.InputLayer(3, sparse_shape.tolist())
        self.voxel_output_shape = output_shape
        middle_layers = []

        num_filters = [num_input_features] + num_filters_down1
        # num_filters = [64] + num_filters_down1
        filters_pairs_d1 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]

        for i, o in filters_pairs_d1:
            middle_layers.append(scn.SubmanifoldConvolution(3, i, o, 3, False))
            middle_layers.append(scn.BatchNormReLU(o, eps=1e-3, momentum=0.99))
        middle_layers.append(
            scn.Convolution(
                3,
                num_filters[-1],
                num_filters[-1], (3, 1, 1), (2, 1, 1),
                bias=False))
        middle_layers.append(
            scn.BatchNormReLU(num_filters[-1], eps=1e-3, momentum=0.99))
        # assert len(num_filters_down2) > 0
        if len(num_filters_down1) == 0:
            num_filters = [num_filters[-1]] + num_filters_down2
        else:
            num_filters = [num_filters_down1[-1]] + num_filters_down2
        filters_pairs_d2 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]
        for i, o in filters_pairs_d2:
            middle_layers.append(scn.SubmanifoldConvolution(3, i, o, 3, False))
            middle_layers.append(scn.BatchNormReLU(o, eps=1e-3, momentum=0.99))
        middle_layers.append(
            scn.Convolution(
                3,
                num_filters[-1],
                num_filters[-1], (3, 1, 1), (2, 1, 1),
                bias=False))
        middle_layers.append(
            scn.BatchNormReLU(num_filters[-1], eps=1e-3, momentum=0.99))
        middle_layers.append(scn.SparseToDense(3, num_filters[-1]))
        self.middle_conv = Sequential(*middle_layers)

    def forward(self, voxel_features, coors, batch_size):
        # coors[:, 1] += 1
        coors = coors.int()[:, [1, 2, 3, 0]]
        ret = self.scn_input((coors.cpu(), voxel_features, batch_size))
        ret = self.middle_conv(ret)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)
        return ret


class ZeroPad3d(nn.ConstantPad3d):
    def __init__(self, padding):
        super(ZeroPad3d, self).__init__(padding, 0)


class MiddleExtractor(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='MiddleExtractor'):
        super(MiddleExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm3d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm3d)
            # BatchNorm3d = change_default_args(
            #     group=32, eps=1e-3, momentum=0.01)(GroupBatchNorm3d)
            Conv3d = change_default_args(bias=False)(nn.Conv3d)
        else:
            BatchNorm3d = Empty
            Conv3d = change_default_args(bias=True)(nn.Conv3d)
        self.voxel_output_shape = output_shape
        self.middle_conv = Sequential(
            ZeroPad3d(1),
            Conv3d(num_input_features, 64, 3, stride=(2, 1, 1)),
            BatchNorm3d(64),
            nn.ReLU(),
            ZeroPad3d([1, 1, 1, 1, 0, 0]),
            Conv3d(64, 64, 3, stride=1),
            BatchNorm3d(64),
            nn.ReLU(),
            ZeroPad3d(1),
            Conv3d(64, 64, 3, stride=(2, 1, 1)),
            BatchNorm3d(64),
            nn.ReLU(),
        )

    def forward(self, voxel_features, coors, batch_size):
        output_shape = [batch_size] + self.voxel_output_shape[1:]
        ret = scatter_nd(coors.long(), voxel_features, output_shape)
        # print('scatter_nd fw:', time.time() - t)
        ret = ret.permute(0, 4, 1, 2, 3)
        ret = self.middle_conv(ret)
        N, C, D, H, W = ret.shape
        ret = ret.view(N, C * D, H, W)

        return ret

##################################  original RPN ##################################

# class RPN(nn.Module):
#     def __init__(self,
#                  use_norm=True,
#                  num_class=2,
#                  layer_nums=[3, 5, 5],
#                  layer_strides=[2, 2, 2],
#                  num_filters=[128, 128, 256],
#                  upsample_strides=[1, 2, 4],
#                  num_upsample_filters=[256, 256, 256],
#                  num_input_filters=128,
#                  num_anchor_per_loc=2,
#                  encode_background_as_zeros=True,
#                  use_direction_classifier=True,
#                  use_groupnorm=False,
#                  num_groups=32,
#                  use_bev=False,
#                  box_code_size=7,
#                  name='rpn'):
#         super(RPN, self).__init__()
#         self._num_anchor_per_loc = num_anchor_per_loc
#         self._use_direction_classifier = use_direction_classifier
#         self._use_bev = use_bev
#         assert len(layer_nums) == 3
#         assert len(layer_strides) == len(layer_nums)
#         assert len(num_filters) == len(layer_nums)
#         assert len(upsample_strides) == len(layer_nums)
#         assert len(num_upsample_filters) == len(layer_nums)
#         factors = []
#         for i in range(len(layer_nums)):
#             assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
#             factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
#         assert all([x == factors[0] for x in factors])
#         if use_norm:
#             if use_groupnorm:
#                 BatchNorm2d = change_default_args(
#                     num_groups=num_groups, eps=1e-3)(GroupNorm)
#             else:
#                 BatchNorm2d = change_default_args(
#                     eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
#             Conv2d = change_default_args(bias=False)(nn.Conv2d)
#             ConvTranspose2d = change_default_args(bias=False)(
#                 nn.ConvTranspose2d)
#         else:
#             BatchNorm2d = Empty
#             Conv2d = change_default_args(bias=True)(nn.Conv2d)
#             ConvTranspose2d = change_default_args(bias=True)(
#                 nn.ConvTranspose2d)

#         # note that when stride > 1, conv2d with same padding isn't
#         # equal to pad-conv2d. we should use pad-conv2d.
#         block2_input_filters = num_filters[0]
#         if use_bev:
#             self.bev_extractor = Sequential(
#                 Conv2d(6, 32, 3, padding=1),
#                 BatchNorm2d(32),
#                 nn.ReLU(),
#                 # nn.MaxPool2d(2, 2),
#                 Conv2d(32, 64, 3, padding=1),
#                 BatchNorm2d(64),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2, 2),
#             )
#             block2_input_filters += 64

#         self.block1 = Sequential(
#             nn.ZeroPad2d(1),
#             Conv2d(
#                 num_input_filters, num_filters[0], 3, stride=layer_strides[0]),
#             BatchNorm2d(num_filters[0]),
#             nn.ReLU(),
#         )
#         for i in range(layer_nums[0]):
#             self.block1.add(
#                 Conv2d(num_filters[0], num_filters[0], 3, padding=1))
#             self.block1.add(BatchNorm2d(num_filters[0]))
#             self.block1.add(nn.ReLU())
#         self.deconv1 = Sequential(
#             ConvTranspose2d(
#                 num_filters[0],
#                 num_upsample_filters[0],
#                 upsample_strides[0],
#                 stride=upsample_strides[0]),
#             BatchNorm2d(num_upsample_filters[0]),
#             nn.ReLU(),
#         )
#         self.block2 = Sequential(
#             nn.ZeroPad2d(1),
#             Conv2d(
#                 block2_input_filters,
#                 num_filters[1],
#                 3,
#                 stride=layer_strides[1]),
#             BatchNorm2d(num_filters[1]),
#             nn.ReLU(),
#         )
#         for i in range(layer_nums[1]):
#             self.block2.add(
#                 Conv2d(num_filters[1], num_filters[1], 3, padding=1))
#             self.block2.add(BatchNorm2d(num_filters[1]))
#             self.block2.add(nn.ReLU())
#         self.deconv2 = Sequential(
#             ConvTranspose2d(
#                 num_filters[1],
#                 num_upsample_filters[1],
#                 upsample_strides[1],
#                 stride=upsample_strides[1]),
#             BatchNorm2d(num_upsample_filters[1]),
#             nn.ReLU(),
#         )
#         self.block3 = Sequential(
#             nn.ZeroPad2d(1),
#             Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
#             BatchNorm2d(num_filters[2]),
#             nn.ReLU(),
#         )
#         for i in range(layer_nums[2]):
#             self.block3.add(
#                 Conv2d(num_filters[2], num_filters[2], 3, padding=1))
#             self.block3.add(BatchNorm2d(num_filters[2]))
#             self.block3.add(nn.ReLU())
#         self.deconv3 = Sequential(
#             ConvTranspose2d(
#                 num_filters[2],
#                 num_upsample_filters[2],
#                 upsample_strides[2],
#                 stride=upsample_strides[2]),
#             BatchNorm2d(num_upsample_filters[2]),
#             nn.ReLU(),
#         )
#         if encode_background_as_zeros:
#             num_cls = num_anchor_per_loc * num_class
#         else:
#             num_cls = num_anchor_per_loc * (num_class + 1)
#         self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> multi scale feature extraction >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        
#         # self.conv_box = nn.Conv2d(
#         #     sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        
#         # self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        
#         # if use_direction_classifier:
#         #     self.conv_dir_cls = nn.Conv2d(
#         #         sum(num_upsample_filters), num_anchor_per_loc * 2, 1)
# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< multi scale feature extraction <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> single scale feature extraction >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        
#         self.conv_box = nn.Conv2d(
#             int(sum(num_upsample_filters)*2/3), num_anchor_per_loc * box_code_size, 1)
        
#         self.conv_cls = nn.Conv2d(int(sum(num_upsample_filters)*2/3), num_cls, 1)
        
#         if use_direction_classifier:
#             self.conv_dir_cls = nn.Conv2d(
#                 int(sum(num_upsample_filters)*2/3), num_anchor_per_loc * 2, 1)
# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< single scale feature extraction <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
           
#         # 新增distance aware
        

#     def forward(self, x, bev=None):
#         x = self.block1(x)
#         # up1 = self.deconv1(x)
#         if self._use_bev:
#             bev[:, -1] = torch.clamp(
#                 torch.log(1 + bev[:, -1]) / np.log(16.0), max=1.0)
#             x = torch.cat([x, self.bev_extractor(bev)], dim=1)
            
#         x = self.block2(x)
#         up2 = self.deconv2(x)
#         x = self.block3(x)
#         up3 = self.deconv3(x)
        
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> multi scale feature extraction >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        
#         # x = torch.cat([up1, up2, up3], dim=1) # [B,C,W,D] [2, 384, 248, 216]
               
#         # box_preds = self.conv_box(x) # [B,C,W,D] [2, 14, 248, 216]
#         # cls_preds = self.conv_cls(x) # [B,C,W,D] [2, 2, 248, 216]
# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< multi scale feature extraction <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> single scale feature extraction >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   
#         x = torch.cat([up2, up3], dim=1)     
#         box_preds = self.conv_box(x) # [B,C,W,D] [2, 14, 248, 216]
#         cls_preds = self.conv_cls(x) # [B,C,W,D] [2, 2, 248, 216]
# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< single scale feature extraction <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
#         # [N, C, y(H), x(W)]
#         box_preds = box_preds.permute(0, 2, 3, 1).contiguous()#[B, y(H), x(W), C]
#         cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
#         ret_dict = {
#             "box_preds": box_preds,
#             "cls_preds": cls_preds,
#         }
        
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> multi scale feature extraction >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#         # if self._use_direction_classifier:
#         #     dir_cls_preds = self.conv_dir_cls(x)
#         #     dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
#         #     ret_dict["dir_cls_preds"] = dir_cls_preds
# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< multi scale feature extraction <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
# #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> single scale feature extraction >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#         if self._use_direction_classifier:
#             dir_cls_preds = self.conv_dir_cls(x)
#             dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
#             ret_dict["dir_cls_preds"] = dir_cls_preds
# #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< single scale feature extraction <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<      
#         return ret_dict

##################################  original RPN ##################################

class DB_block(nn.Module):
    def __init__(self, current_channel=[128, 128, 128]):
        super(DB_block, self).__init__()
        
        self.DB_down = Sequential(
            nn.Conv2d(current_channel[0], current_channel[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(current_channel[0]),
            nn.ReLU(),
            nn.Conv2d(current_channel[0], current_channel[1], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(current_channel[1]),
            nn.ReLU(),
            nn.Conv2d(current_channel[1], current_channel[2], 3, stride=1, padding=1)
        )
        
        self.DB_up = nn.Conv2d(current_channel[0], current_channel[2], 1, stride=1)
        
        self.bn = nn.BatchNorm2d(current_channel[2])
        
            
    def forward(self, x):
        
        x_skip = self.DB_up(x)
        x = self.DB_down(x)
        x += x_skip
        x = F.relu(x)
        x = self.bn(x)
        x_half = F.avg_pool2d(x, 2)
        return x, x_half
    
class UB_block(nn.Module):
    def __init__(self, current_channel=[128, 128]):
        super(UB_block, self).__init__()
        
        self.UB = nn.ConvTranspose2d(
                current_channel[0],
                current_channel[1],
                2,
                stride=2)
        
        self.fuse = Sequential(
            nn.Conv2d(current_channel[1]*2, current_channel[1], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(current_channel[1]),
            nn.ReLU()
        )
        
            
    def forward(self, x, y):
        
        x = self.UB(x)
        x = torch.cat([x, y], dim=1)
        x = self.fuse(x)
        return x

class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 num_filters=[128, 128, 256],
                 upsample_strides=[1, 2, 4],
                 num_upsample_filters=[256, 256, 256],
                 num_input_filters=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_bev=False,
                 box_code_size=7,
                 name='rpn',
                 DARN=True,
                 DARN_method='p',
                 DARN_order=7,
                 DARN_cnn=1,
                 p_attention=True,
                 rank=True):
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self.DARN = DARN
        self.p_attention = p_attention
        self.rank=rank
        
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        if use_bev:
            pass
            self.bev_extractor = Sequential(
                Conv2d(6, 32, 3, padding=1),
                BatchNorm2d(32),
                nn.ReLU(),
                # nn.MaxPool2d(2, 2),
                Conv2d(32, 64, 3, padding=1),
                BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
            )
            block2_input_filters += 64

        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_filters, num_filters[0], 3, stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
            # nn.LeakyReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
            # self.block1.add(nn.LeakyReLU())
            
        if not DARN:
            self.deconv1 = Sequential(
                ConvTranspose2d(
                    num_filters[0],
                    num_upsample_filters[0],
                    upsample_strides[0],
                    stride=upsample_strides[0]),
                BatchNorm2d(num_upsample_filters[0]),
                nn.ReLU(),
            )
        
        
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
            # nn.LeakyReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
            # self.block2.add(nn.LeakyReLU())

            
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
            # nn.LeakyReLU(),
        )
        
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),
            # nn.LeakyReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
            # self.block3.add(nn.LeakyReLU())
        
        
        self.deconv3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]
            ),
            nn.ReLU(),
            # nn.LeakyReLU(),
        )
                
        
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
            
        self.upx2 = nn.Upsample(scale_factor=2, mode='bicubic')
        
        # self.collect_shape = nn.MaxPool2d(2, stride=2)
        
        

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 2D detector setting >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        
        self.conv_box = nn.Conv2d(
            int(sum(num_upsample_filters)), num_anchor_per_loc * box_code_size, 1)
        
        self.conv_cls = nn.Conv2d(int(sum(num_upsample_filters)), num_cls, 1)
        
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                int(sum(num_upsample_filters)), num_anchor_per_loc * 2, 1)
            
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 2D detector setting <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 新增distance aware >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
        if DARN:
            if DARN_method=='p':
                if DARN_cnn == 1:                 
                    self.coefficient_pridict = torch.nn.Conv2d(256, 7, (1,1), (1,1))
                if DARN_cnn == 3:                 
                    self.coefficient_pridict = torch.nn.Conv2d(256, 7, (3,3), (1,1), 1)
            
            elif DARN_method=='f':
                self.coefficient_pridict = torch.nn.Conv2d(256, 1, (1,1), (1,1))
            
            self.coefficient_sofmax = torch.nn.Softmax(dim=1)
            self.coefficient_sigmoid = torch.nn.Sigmoid()
            
            # for car
            self.temp_ratio = (np.arange(216.0, dtype="float32")/216.0)[np.newaxis,:]
            
            # # # for person
            # self.temp_ratio = (np.arange(296.0, dtype="float32")/296.0)[np.newaxis,:]
            
            if DARN_order==5:
                self.temp_ratio = np.concatenate((np.power(self.temp_ratio, 5), np.power(self.temp_ratio, 4), \
                                                  np.power(self.temp_ratio, 3), np.power(self.temp_ratio, 2), self.temp_ratio), axis=0)
            
            if DARN_order==7:
                self.temp_ratio = np.concatenate((np.power(self.temp_ratio, 7), np.power(self.temp_ratio, 6), np.power(self.temp_ratio, 5), np.power(self.temp_ratio, 4), \
                                                  np.power(self.temp_ratio, 3), np.power(self.temp_ratio, 2), self.temp_ratio), axis=0)
                
            self.temp_ratio = torch.from_numpy(self.temp_ratio).cuda()
            self.temp_ratio = torch.unsqueeze(torch.unsqueeze(self.temp_ratio, 1),0)

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 新增distance aware <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> insert Dynamic Head (channel attention) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>         
        if p_attention:
            self.dyrelu = DyReLUB(384, conv_type='2d')

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< insert Dynamic Head (channel attention) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


    def forward(self, x, bev=None):

        x = self.block1(x)
        
        if not self.DARN:
            up1 = self.deconv1(x)
        if self._use_bev:
            bev[:, -1] = torch.clamp(
                torch.log(1 + bev[:, -1]) / np.log(16.0), max=1.0)
            x = torch.cat([x, self.bev_extractor(bev)], dim=1)
            
        x = self.block2(x)
        
        up2 = self.deconv2(x)
        x = self.block3(x)
        
        up3 = self.deconv3(x)  
        
        if self.DARN:
            xx = torch.cat([up2, up3], dim=1)       
            combine_coefficient = self.coefficient_pridict(xx)
            combine_coefficient = self.coefficient_sofmax(combine_coefficient)
            combine_ratio = combine_coefficient*self.temp_ratio
            combine_ratio = torch.sum(combine_ratio, 1, keepdim=True)
        
        
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> use mean combine_ratio >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        
        
        # combine_ratio = np.load("/data2/chihjen/second.pytorch/second/mean.npy")

        # if x.shape[0] == 2:
        #     combine_ratio = np.concatenate((combine_ratio, combine_ratio), axis=0)
        
        # combine_ratio = torch.from_numpy(combine_ratio).cuda()
        
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< use mean combine_ration <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<              
        
        ###############################  ratio  no  limit  ver. ###############################
        # combine_ratio = self.coefficient_pridict(xx)
        # combine_ratio = self.coefficient_sigmoid(combine_ratio)


        if self.DARN:
            distance_aware_feature = up3*(1-combine_ratio) + up2*combine_ratio # [B,C,W,D] [2, 128, 248, 216]
        
        # distance_aware_feature = up2*(1-combine_ratio) + up1*combine_ratio # [B,C,W,D] 
  
        if not self.DARN:
            x = torch.cat([up1, up2, up3], dim=1)  
        
        # x = torch.cat([up1, up2, up3, distance_aware_feature], dim=1) 
        
        if self.DARN:
            x = torch.cat([up2, up3, distance_aware_feature], dim=1)

        
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> insert Dynamic Head (channel attention) >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> 
        
        if self.p_attention & self.DARN:
            x_attention = self.dyrelu(x)
            x_attention = torch.cat([x_attention[:, :256, :, :], distance_aware_feature], dim=1)
            
        if self.p_attention & (not self.DARN):
            x_attention = self.dyrelu(x)

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< insert Dynamic Head (channel attention) <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
         

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> remove distance information >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>   
          
        # remove_index = 1
        # # xd = x.view(x.shape[0], x.shape[1], -1)
        # xd = x_attention.view(x.shape[0], x.shape[1], -1)
        # xd = xd.permute(0, 2, 1).contiguous()
        
        # xU, xS, xV = torch.svd(xd[0,:,:])
        # xS[remove_index] = 0
        # x_no_distance = torch.unsqueeze(torch.matmul(xU, torch.matmul(torch.diag(xS), xV.transpose(-2, -1))), 0)
        
        # if x.shape[0] == 2:
        #     xU, xS, xV = torch.svd(xd[1,:,:])
        #     xS[remove_index] = 0
        #     xd2 = torch.unsqueeze(torch.matmul(xU, torch.matmul(torch.diag(xS), xV.transpose(-2, -1))), 0)
        #     x_no_distance = torch.cat([x_no_distance, xd2], 0)
            
        # x_no_distance = x_no_distance.permute(0, 2, 1).contiguous()
        # x_no_distance = x_no_distance.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        
        # box_preds = self.conv_box(x_no_distance) # [B,C,W,D] [2, 14, 248, 216]
        # cls_preds = self.conv_cls(x_no_distance)# [B,C,W,D] [2, 2, 248, 216]
            
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< remove distance information <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 
        
        if self.p_attention:
            box_preds = self.conv_box(x_attention) # [B,C,W,D] [2, 14, 248, 216]
            cls_preds = self.conv_cls(x_attention)# [B,C,W,D] [2, 2, 248, 216]
        
        if not self.p_attention:
            box_preds = self.conv_box(x) # [B,C,W,D] [2, 14, 248, 216]
            cls_preds = self.conv_cls(x)# [B,C,W,D] [2, 2, 248, 216]
            
        
        
        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()#[B, y(H), x(W), C]
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()

        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }


        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
                     
            
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
            

        # return ret_dict, combine_ratio#, combine_coefficient3
        
        if self.rank:
            return ret_dict, self.upx2(distance_aware_feature)
        if not self.rank:
            return ret_dict
        
        # # for person
        # return ret_dict, distance_aware_feature#, upx2(up2), upx2(up3)

class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives" 
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"


class VoxelNet(nn.Module):
    def __init__(self,
                 output_shape,
                 num_class=2,
                 num_input_features=4,
                 vfe_class_name="VoxelFeatureExtractor",
                 vfe_num_filters=[32, 128],
                 with_distance=False,
                 middle_class_name="SparseMiddleExtractor",
                 middle_num_filters_d1=[64],
                 middle_num_filters_d2=[64, 64],
                 rpn_class_name="RPN",
                 rpn_layer_nums=[3, 5, 5],
                 rpn_layer_strides=[2, 2, 2],
                 rpn_num_filters=[128, 128, 256],
                 rpn_upsample_strides=[1, 2, 4],
                 rpn_num_upsample_filters=[256, 256, 256],
                 use_norm=True,
                 use_groupnorm=False,
                 num_groups=32,
                 use_sparse_rpn=False,
                 use_direction_classifier=True,
                 use_sigmoid_score=False,
                 encode_background_as_zeros=True,
                 use_rotate_nms=True,
                 multiclass_nms=False,
                 nms_score_threshold=0.5,
                 nms_pre_max_size=1000,
                 nms_post_max_size=20,
                 nms_iou_threshold=0.1,
                 target_assigner=None,
                 use_bev=False,
                 lidar_only=False,
                 cls_loss_weight=1.0,
                 loc_loss_weight=1.0,
                 pos_cls_weight=1.0,
                 neg_cls_weight=1.0,
                 direction_loss_weight=1.0,
                 loss_norm_type=LossNormType.NormByNumPositives,
                 encode_rad_error_by_sin=False,
                 loc_loss_ftor=None,
                 cls_loss_ftor=None,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1),
                 name='voxelnet',
                 SVDnet=None):
        super().__init__()
        self.name = name
        self._num_class = num_class
        self._use_rotate_nms = use_rotate_nms
        self._multiclass_nms = multiclass_nms
        self._nms_score_threshold = nms_score_threshold
        self._nms_pre_max_size = nms_pre_max_size
        self._nms_post_max_size = nms_post_max_size
        self._nms_iou_threshold = nms_iou_threshold
        self._use_sigmoid_score = use_sigmoid_score
        self._encode_background_as_zeros = encode_background_as_zeros
        self._use_sparse_rpn = use_sparse_rpn
        self._use_direction_classifier = use_direction_classifier
        self._use_bev = use_bev
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0
        self._num_input_features = num_input_features
        self._box_coder = target_assigner.box_coder
        self._lidar_only = lidar_only
        self.target_assigner = target_assigner
        self._pos_cls_weight = pos_cls_weight
        self._neg_cls_weight = neg_cls_weight
        self._encode_rad_error_by_sin = encode_rad_error_by_sin
        self._loss_norm_type = loss_norm_type
        self._dir_loss_ftor = WeightedSoftmaxClassificationLoss()

        self._loc_loss_ftor = loc_loss_ftor
        self._cls_loss_ftor = cls_loss_ftor
        self._direction_loss_weight = direction_loss_weight
        self._cls_loss_weight = cls_loss_weight
        self._loc_loss_weight = loc_loss_weight
        
        self.rank = SVDnet["rank"]
        self.L2 = SVDnet["L2"]

        vfe_class_dict = {
            "VoxelFeatureExtractor": VoxelFeatureExtractor,
            "VoxelFeatureExtractorV2": VoxelFeatureExtractorV2,
            "PillarFeatureNet": PillarFeatureNet
        }
        vfe_class = vfe_class_dict[vfe_class_name]
        if vfe_class_name == "PillarFeatureNet":
            self.voxel_feature_extractor = vfe_class(
                num_input_features,
                use_norm,
                num_filters=vfe_num_filters,
                with_distance=with_distance,
                voxel_size=voxel_size,
                pc_range=pc_range
            )
        else:
            self.voxel_feature_extractor = vfe_class(
                num_input_features,
                use_norm,
                num_filters=vfe_num_filters,
                with_distance=with_distance)

        print("middle_class_name", middle_class_name)
        if middle_class_name == "PointPillarsScatter":
            self.middle_feature_extractor = PointPillarsScatter(output_shape=output_shape,
                                                                num_input_features=vfe_num_filters[-1])
            num_rpn_input_filters = self.middle_feature_extractor.nchannels
        else:
            mid_class_dict = {
                "MiddleExtractor": MiddleExtractor,
                "SparseMiddleExtractor": SparseMiddleExtractor,
            }
            mid_class = mid_class_dict[middle_class_name]
            self.middle_feature_extractor = mid_class(
                output_shape,
                use_norm,
                num_input_features=vfe_num_filters[-1],
                num_filters_down1=middle_num_filters_d1,
                num_filters_down2=middle_num_filters_d2)
            if len(middle_num_filters_d2) == 0:
                if len(middle_num_filters_d1) == 0:
                    num_rpn_input_filters = int(vfe_num_filters[-1] * 2)
                else:
                    num_rpn_input_filters = int(middle_num_filters_d1[-1] * 2)
            else:
                num_rpn_input_filters = int(middle_num_filters_d2[-1] * 2)

        rpn_class_dict = {
            "RPN": RPN,
        }
        rpn_class = rpn_class_dict[rpn_class_name]
        self.rpn = rpn_class(
            use_norm=True,
            num_class=num_class,
            layer_nums=rpn_layer_nums,
            layer_strides=rpn_layer_strides,
            num_filters=rpn_num_filters,
            upsample_strides=rpn_upsample_strides,
            num_upsample_filters=rpn_num_upsample_filters,
            num_input_filters=num_rpn_input_filters,
            num_anchor_per_loc=target_assigner.num_anchors_per_location,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_bev=use_bev,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=target_assigner.box_coder.code_size,
            DARN=SVDnet["DARN"],
            DARN_method=SVDnet["DARN_method"],
            DARN_order=SVDnet["DARN_order"],
            DARN_cnn=SVDnet["DARN_cnn"],
            p_attention=SVDnet["p_attention"],
            rank=SVDnet["rank"])

        self.rpn_acc = metrics.Accuracy(
            dim=-1, encode_background_as_zeros=encode_background_as_zeros)
        self.rpn_precision = metrics.Precision(dim=-1)
        self.rpn_recall = metrics.Recall(dim=-1)
        self.rpn_metrics = metrics.PrecisionRecall(
            dim=-1,
            thresholds=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
            use_sigmoid_score=use_sigmoid_score,
            encode_background_as_zeros=encode_background_as_zeros)

        self.rpn_cls_loss = metrics.Scalar()
        self.rpn_loc_loss = metrics.Scalar()
        self.rpn_total_loss = metrics.Scalar()
        self.register_buffer("global_step", torch.LongTensor(1).zero_())

    def update_global_step(self):
        self.global_step += 1

    def get_global_step(self):
        return int(self.global_step.cpu().numpy()[0])

    def forward(self, example):
        """module's forward should always accept dict and return loss.
        """
        voxels = example["voxels"]
        num_points = example["num_points"]
        coors = example["coordinates"]
        batch_anchors = example["anchors"]
        batch_size_dev = batch_anchors.shape[0]
        t = time.time()
        # features: [num_voxels, max_num_points_per_voxel, 7]
        # num_points: [num_voxels]
        # coors: [num_voxels, 4]
        voxel_features = self.voxel_feature_extractor(voxels, num_points, coors)
        # print("voxel: ", voxels.shape)
        # print("num_points: ", num_points.shape)
        

        if self._use_sparse_rpn:
            preds_dict = self.sparse_rpn(voxel_features, coors, batch_size_dev)
        else:
            spatial_features = self.middle_feature_extractor(
                voxel_features, coors, batch_size_dev)
            
            # check_spatial_features = np.max(spatial_features[0,:,:,:].detach().cpu().numpy(), axis=0)
            # sizee, _ = np.where(check_spatial_features == 0)
            # print(sizee.shape)

            if self._use_bev:
                preds_dict = self.rpn(spatial_features, example["bev_map"])
            else:
                if not self.rank:
                    preds_dict = self.rpn(spatial_features)
                if self.rank:
                    preds_dict, feature_d = self.rpn(spatial_features)
                # preds_dict, feature_before, feature_after = self.rpn(spatial_features)
                # preds_dict, feature_d, extra_losses = self.rpn(spatial_features)
                # preds_dict, feature_d = self.rpn(spatial_features)


        # preds_dict["voxel_features"] = voxel_features
        # preds_dict["spatial_features"] = spatial_features
        
        ###########################################################################################################################
        example_index = example["image_idx"]
        ###########################################################################################################################
        # example_count = 0
        # for example_id in example_index:
 
        #     np.save("/data2/chihjen/second.pytorch/second/combine_ratio/" + \
        #                           "%06d.npy" % example_id, feature_d[example_count,0,:,:].cpu().detach().numpy())
        #     # np.save("/data2/chihjen/second.pytorch/second/residual_ratio_3/" + \
        #     #                       "%06d.npy" % example_id, combine_ratio3[example_count,0,:,:].cpu().detach().numpy())
                
        #     example_count = example_count + 1
        
        
        box_preds = preds_dict["box_preds"] #[B, y(H), x(W), C] [2, 248, 216, 14]
        cls_preds = preds_dict["cls_preds"] #[B, y(H), x(W), C]
        

        self._total_forward_time += time.time() - t
        if self.training:
            labels = example['labels']
            reg_targets = example['reg_targets']

            cls_weights, reg_weights, cared = prepare_loss_weights(
                labels,
                pos_cls_weight=self._pos_cls_weight,
                neg_cls_weight=self._neg_cls_weight,
                loss_norm_type=self._loss_norm_type,
                dtype=voxels.dtype)
            cls_targets = labels * cared.type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)
            
            # loc_loss shape [batch, 248*216*2, 7]
            loc_loss, cls_loss = create_loss(
                self._loc_loss_ftor,
                self._cls_loss_ftor,
                box_preds=box_preds,
                cls_preds=cls_preds,
                cls_targets=cls_targets,
                cls_weights=cls_weights,
                reg_targets=reg_targets,
                reg_weights=reg_weights,
                num_class=self._num_class,
                encode_rad_error_by_sin=self._encode_rad_error_by_sin,
                encode_background_as_zeros=self._encode_background_as_zeros,
                box_code_size=self._box_coder.code_size,
            )
            
            loc_loss_reduced = loc_loss.sum() / batch_size_dev
            loc_loss_reduced *= self._loc_loss_weight #1
            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
            cls_pos_loss /= self._pos_cls_weight
            cls_neg_loss /= self._neg_cls_weight
            cls_loss_reduced = cls_loss.sum() / batch_size_dev
            cls_loss_reduced *= self._cls_loss_weight #2 
        
            
            #################################### rank loss person  ####################################
            
            # example_index = example["image_idx"]
            # rank_loss = [0.0]
            # for spatial_features_index in range(spatial_features.shape[0]):
            #     # print("spatial_features : ", spatial_features[spatial_features_index,:,:,:].permute(1,2,0).shape)
            #     foreground_grid_num = \
            #         np.load("/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object/training/label_grid_num_person/" + \
            #                   "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)
            #     foreground_grid_flag = torch.from_numpy((np.squeeze(foreground_grid_num) > 0) & (np.squeeze(foreground_grid_num) < 20)).cuda()
            #     # print("foreground_grid_flag : ", foreground_grid_flag.shape)
            #     # print(spatial_features[spatial_features_index,:,:,:].permute(1,2,0)[foreground_grid_flag].shape)
            #     svd_matrix = spatial_features[spatial_features_index,:,:,:].permute(1,2,0)[foreground_grid_flag].cuda()
                
            #     if svd_matrix.shape[0] <= 1:
            #         pass
            #     else:
            #         try:
            #             _, s, _ = torch.svd(svd_matrix)
                        
            #             if s[1] == 0:
            #                 # print("s[0]", s[0])
            #                 # print("s[1]", s[1])
            #                 pass
            #             else:
                                
            #                 # print("s[0]", s[0])
            #                 # print("s[1]", s[1])
            #                 # print(s[0]/s[1] + 1)
            #                 if rank_loss[0] == 0:
            #                     rank_loss.pop()
            #                 rank_loss.append(F.sigmoid(s[0]/s[1]) - 0.75)
                          
            #         except:
            #             pass
            #             # _, s, _ = torch.svd(svd_matrix + 1e-4*svd_matrix.mean()*(torch.rand(svd_matrix.shape).cuda()))
                    
            #         # _, s, _ = torch.svd(svd_matrix)
            #         # print(rank[0]/rank[1] + 1)
            #         # print(rank)
            #         # rank_loss += s[0]/s[1] + 1
            # loss_add = min(rank_loss)*(-1)
            
            # rank_loss2 = [0.0]
            # for spatial_features_index in range(feature_d.shape[0]):
            #     # print("spatial_features : ", spatial_features[spatial_features_index,:,:,:].permute(1,2,0).shape)
            #     foreground_grid_num = \
            #         np.load("/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object/training/label_grid_num_person/" + \
            #                   "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)
            #     foreground_grid_flag = torch.from_numpy((np.squeeze(foreground_grid_num) > 0) & (np.squeeze(foreground_grid_num) < 20)).cuda()
            #     # print("foreground_grid_flag : ", foreground_grid_flag.shape)
            #     # print(spatial_features[spatial_features_index,:,:,:].permute(1,2,0)[foreground_grid_flag].shape)
            #     svd_matrix = feature_d[spatial_features_index,:,:,:].permute(1,2,0)[foreground_grid_flag].cuda()
                
            #     if svd_matrix.shape[0] <= 1:
            #         pass
            #     else:
            #         try:
            #             _, s, _ = torch.svd(svd_matrix)
                        
            #             if s[1] == 0:
            #                 # print("s[0]", s[0])
            #                 # print("s[1]", s[1])
            #                 pass
            #             else:
                                
            #                 # print("s[0]", s[0])
            #                 # print("s[1]", s[1])
            #                 # print(s[0]/s[1] + 1)
            #                 if rank_loss2[0] == 0:
            #                     rank_loss2.pop()
            #                 rank_loss2.append(F.sigmoid(s[0]/s[1]) - 0.75)
                          
            #         except:
            #             pass
            #             # _, s, _ = torch.svd(svd_matrix + 1e-4*svd_matrix.mean()*(torch.rand(svd_matrix.shape).cuda()))
                    
            #         # _, s, _ = torch.svd(svd_matrix)
            #         # print(rank[0]/rank[1] + 1)
            #         # print(rank)
            #         # rank_loss += s[0]/s[1] + 1
            # loss_add2 = min(rank_loss2)*(-1)

            
            #################################### rank loss person  ####################################
            
            # #################################### density rank loss  ####################################
            if self.rank:
                example_index = example["image_idx"]
                rank_loss_l = [0.0]
                rank_loss_h = [0.0]
                for spatial_features_index in range(spatial_features.shape[0]):
                    foreground_grid_num = \
                        np.load("/mnt/HDD0/data/evan/KITTI_DATASET_ROOT/training/label_grid_num/" + \
                                  "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)#[:,::-1,::-1]
                        # np.load("/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object/training/label_grid_num_person/" + \
                        #           "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)
                    foreground_grid_flag = torch.from_numpy(np.squeeze(foreground_grid_num) > 0)
                    foreground_grid_flag_h = (foreground_grid_flag & torch.from_numpy(np.squeeze(foreground_grid_num) > 15)).cuda()  # car 15 
                    foreground_grid_flag_l = (foreground_grid_flag & torch.from_numpy(np.squeeze(foreground_grid_num) <= 15)).cuda() # person 5
                    svd_matrix_h = spatial_features[spatial_features_index,:,:,:].permute(1,2,0)[foreground_grid_flag_h].cuda()
                    svd_matrix_l = spatial_features[spatial_features_index,:,:,:].permute(1,2,0)[foreground_grid_flag_l].cuda()        
                    # print(np.sum(foreground_grid_flag_h*1).reshape((-1,1)))
                    # print(np.sum(foreground_grid_flag_l*1).reshape((-1,1)))
                    if svd_matrix_h.shape[0] <= 1:
                        pass
                    else:
                        try:
                            _, s1, _ = torch.svd(svd_matrix_h)
                            if s1[1] == 0:
                                pass
                            else:
                                if rank_loss_h[0] == 0:
                                    rank_loss_h.pop()
                                rank_loss_h.append(F.sigmoid(s1[0]/s1[1])-0.75)
                                # rank_loss_h.append(F.sigmoid(s1[0]/s1[1]))
                        except:
                            pass
                    
                    if svd_matrix_l.shape[0] <= 1:
                        pass
                    else:
                        try:
                            _, s2, _ = torch.svd(svd_matrix_l)
                            if s2[1] == 0:
                                pass
                            else:
                                if rank_loss_l[0] == 0:
                                    rank_loss_l.pop()
                                rank_loss_l.append(F.sigmoid(s2[0]/s2[1])-0.75)
                                # rank_loss_l.append(F.sigmoid(s2[0]/s2[1]))
                        except:
                            pass
                        
                loss_add = (min(rank_loss_l) + min(rank_loss_h))*(-0.5)
                # loss_add = (min(rank_loss_l) + min(rank_loss_h))*(0.2)
            
            # #################################### density rank loss  ####################################
            
            # #################################### density rank loss for feature_distance  ####################################
            if self.rank:
                example_index = example["image_idx"]
                rank_loss_l = [0.0]
                rank_loss_h = [0.0]
                # bg_loss = 0
                # fg_loss = 0
                for spatial_features_index in range(feature_d.shape[0]):
                    # print("spatial_features : ", spatial_features[spatial_features_index,:,:,:].permute(1,2,0).shape)
                    foreground_grid_num = \
                        np.load("/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object/training/label_grid_num/" + \
                                  "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)#[:,::-1,::-1]
                        # np.load("/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object/training/label_grid_num_person/" + \
                        #           "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)
                    foreground_grid_flag = torch.from_numpy(np.squeeze(foreground_grid_num) > 0)
                    foreground_grid_flag_h = (foreground_grid_flag & torch.from_numpy(np.squeeze(foreground_grid_num) > 15)).cuda()  # car 15 
                    foreground_grid_flag_l = (foreground_grid_flag & torch.from_numpy(np.squeeze(foreground_grid_num) <= 15)).cuda() # person 5
                    svd_matrix_h = feature_d[spatial_features_index,:,:,:].permute(1,2,0)[foreground_grid_flag_h].cuda()
                    svd_matrix_l = feature_d[spatial_features_index,:,:,:].permute(1,2,0)[foreground_grid_flag_l].cuda()
                    
                    if svd_matrix_h.shape[0] <= 1:
                        pass
                    else:
                        try:
                            _, s1, _ = torch.svd(svd_matrix_h)
                            if s1[1] == 0:
                                pass
                            else:
                                if rank_loss_h[0] == 0:
                                    rank_loss_h.pop()
                                rank_loss_h.append(F.sigmoid(s1[0]/s1[1])-0.75)
                                # rank_loss_h.append(F.sigmoid(s1[0]/s1[1]))
                        except:
                            pass
                    
                    if svd_matrix_l.shape[0] <= 1:
                        pass
                    else:
                        try:
                            _, s2, _ = torch.svd(svd_matrix_l)
                            if s2[1] == 0:
                                pass
                            else:
                                if rank_loss_l[0] == 0:
                                    rank_loss_l.pop()
                                rank_loss_l.append(F.sigmoid(s2[0]/s2[1])-0.75)
                                # rank_loss_l.append(F.sigmoid(s2[0]/s2[1]))
                        except:
                            pass
    
                loss_add2 = (min(rank_loss_l) + min(rank_loss_h))*(-0.5) # original 0.5
                # loss_add2 = (min(rank_loss_l) + min(rank_loss_h))*(0.2)
            
            # #################################### density rank loss for feature_distance  ####################################
            
            
            #################################### L2  ####################################
            if self.L2:
                example_index = example["image_idx"]
                for spatial_features_index in range(spatial_features.shape[0]):
                    foreground_grid_num = \
                        np.load("/mnt/HDD0/data/evan/KITTI_DATASET_ROOT/training/label_grid_num/" + \
                                  "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)#[:,::-1,::-1]
                    foreground_grid_flag = torch.from_numpy(np.squeeze(foreground_grid_num) > 0).cuda()
                    foreground_matrix = spatial_features[spatial_features_index,:,:,:].permute(1,2,0)[foreground_grid_flag].cuda()    
                    
                    base_n = foreground_matrix.shape[0]
                    
                    if base_n <= 1:
                        loss_add = 0
                    else:
    
                        base_i = random.randint(0, base_n - 1)
                        base_f = foreground_matrix[base_i,:].repeat(base_n, 1)
                        base_r = torch.sum(torch.abs(base_f - foreground_matrix)) / (base_n - 1)
                        loss_add = F.sigmoid(base_r*base_r)*0.1
                    
                loss_add2 = 0
            #################################### L2  ####################################
            
            if (not self.L2) & (not self.rank):
                loss_add = 0
                loss_add2 = 0
            
            loss = loc_loss_reduced + cls_loss_reduced + loss_add + loss_add2
                
            extra_losses_reduce = {
                # "box_preds_up2": loc_loss_reduced_up2,
                # "cls_preds_up2": cls_loss_reduced_up2,
                # "box_preds_up3": loc_loss_reduced_up3,
                # "cls_preds_up3": cls_loss_reduced_up3
                }    
            

            if self._use_direction_classifier:
                dir_targets = get_direction_target(example['anchors'],
                                                   reg_targets)
                dir_logits = preds_dict["dir_cls_preds"].view(
                    batch_size_dev, -1, 2)
                weights = (labels > 0).type_as(dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self._dir_loss_ftor(
                    dir_logits, dir_targets, weights=weights)
                dir_loss = dir_loss.sum() / batch_size_dev
                loss += dir_loss * self._direction_loss_weight

            return {
                "loss": loss,
                "cls_loss": cls_loss,
                "loc_loss": loc_loss,
                "cls_pos_loss": cls_pos_loss,
                "cls_neg_loss": cls_neg_loss,
                "cls_preds": cls_preds,
                "dir_loss_reduced": dir_loss,
                "cls_loss_reduced": cls_loss_reduced,
                "loc_loss_reduced": loc_loss_reduced,
                "cared": cared,
                "loss_rank": loss_add,
                "df_rank_loss": loss_add2,
                "extra_loss": extra_losses_reduce,
            }
        else:
            
            
            ##################################  pillar feature output before 2d extract  #########################################
            
            # # example_index = example["image_idx"]
            # for spatial_features_index in range(spatial_features.shape[0]):
            #     # print("spatial_features : ", spatial_features[spatial_features_index,:,:,:].permute(1,2,0).shape)
                
            #     # for car
            #     foreground_grid_num = \
            #         np.load("/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object/training/label_grid_num/" + \
            #                   "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)[:,::-1,::-1]
                        
            #     # for people
            #     # foreground_grid_num = \
            #     #     np.load("/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object/training/label_grid_num_person/" + \
            #     #               "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)
                
            #     # grid distance
            #     _, grid_distance = np.mgrid[0:foreground_grid_num.shape[1], 0:foreground_grid_num.shape[2]]
            #     grid_distance = foreground_grid_num.shape[2] - grid_distance
                
            #     grid_distance = grid_distance[::-1,::-1]
                        
            #     foreground_grid_flag = torch.from_numpy(np.squeeze(foreground_grid_num) > 0).cuda()
            #     foreground_grid_num_column = torch.from_numpy(np.squeeze(foreground_grid_num.copy()))[foreground_grid_flag].cuda()
            #     foreground_grid_distance = torch.from_numpy(grid_distance.copy())[foreground_grid_flag].cuda()
            #     background_grid_distance = torch.from_numpy(grid_distance.copy())[~foreground_grid_flag].cuda()

            #     foreground_spatial_features = spatial_features[spatial_features_index,:,:,:].permute(1,2,0)[foreground_grid_flag].cuda()
            #     background_spatial_features = spatial_features[spatial_features_index,:,:,:].permute(1,2,0)[~foreground_grid_flag].cuda()
                
            #     check_foreground_num = foreground_grid_num_column.shape[0]
            #     if check_foreground_num == 0:
            #         pass
            #     else:
            #         foreground_grid_num_column = np.reshape(foreground_grid_num_column.cpu().detach().numpy(),(check_foreground_num,1))
            #         foreground_spatial_features = foreground_spatial_features.cpu().detach().numpy()
            #         foreground_grid_distance = np.reshape(foreground_grid_distance.cpu().detach().numpy(),(check_foreground_num,1))
                    
            #         # output fg feature 
            #         output_pillar_feature = np.concatenate((foreground_spatial_features, foreground_grid_num_column, foreground_grid_distance), axis=1)     
                    
            #         # # output bg feature 
            #         # bg_select = torch.randperm(background_spatial_features.shape[0])[:check_foreground_num*3]
            #         # background_spatial_features = background_spatial_features[bg_select].cpu().detach().numpy()
            #         # background_grid_distance = background_grid_distance[bg_select].cpu().detach().numpy()[:,np.newaxis]
            #         # label_fg_bg = np.concatenate((np.ones((check_foreground_num,1)),np.zeros((background_spatial_features.shape[0],1))), axis=0)   
            #         # d_fg_bg = np.concatenate((foreground_grid_distance,background_grid_distance), axis=0)
                                
            #         # output_pillar_feature = np.concatenate((foreground_spatial_features, background_spatial_features), axis=0)
            #         # output_pillar_feature = np.concatenate((output_pillar_feature, label_fg_bg, d_fg_bg), axis=1)
                    
            #         np.save("/data2/chihjen/second.pytorch/second/pytorch/pseudo_image_rank/" + \
            #                   "%06d.npy" % example_index[spatial_features_index], output_pillar_feature)
                
            # ##################################  pillar feature output before 2d extract  #########################################
            
            # ##################################  pillar feature output after 2d extract  #########################################
            
            # for spatial_features_index in range(feature_d.shape[0]):                
            #     # for car
            #     foreground_grid_num = \
            #         np.load("/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object/training/label_grid_num/" + \
            #                   "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)[:,::-1,::-1]        
            #     # for people
            #     # foreground_grid_num = \
            #     #     np.load("/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object/training/label_grid_num_person/" + \
            #     #               "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)
                
            #     # grid distance
            #     _, grid_distance = np.mgrid[0:foreground_grid_num.shape[1], 0:foreground_grid_num.shape[2]]
            #     grid_distance = foreground_grid_num.shape[2] - grid_distance
                
            #     grid_distance = grid_distance[::-1,::-1]
                
            #     foreground_grid_flag = torch.from_numpy(np.squeeze(foreground_grid_num) > 0).cuda()
            #     foreground_grid_num_column = torch.from_numpy(np.squeeze(foreground_grid_num.copy()))[foreground_grid_flag].cuda()
            #     foreground_grid_distance = torch.from_numpy(grid_distance.copy())[foreground_grid_flag].cuda()
            #     background_grid_distance = torch.from_numpy(grid_distance.copy())[~foreground_grid_flag].cuda()
                
            #     foreground_spatial_features = feature_d[spatial_features_index,:,:,:].permute(1,2,0)[foreground_grid_flag].cuda()
            #     background_spatial_features = feature_d[spatial_features_index,:,:,:].permute(1,2,0)[~foreground_grid_flag].cuda()
                
            #     check_foreground_num = foreground_grid_num_column.shape[0]
            #     if check_foreground_num == 0:
            #         pass
            #     else:
            #         # output fg feature
            #         foreground_grid_num_column = np.reshape(foreground_grid_num_column.cpu().detach().numpy(),(check_foreground_num,1))
            #         foreground_spatial_features = foreground_spatial_features.cpu().detach().numpy()
            #         foreground_grid_distance = np.reshape(foreground_grid_distance.cpu().detach().numpy(),(check_foreground_num,1))
            #         output_pillar_feature = np.concatenate((foreground_spatial_features, foreground_grid_num_column, foreground_grid_distance), axis=1)
                    
            #         np.save("/data2/chihjen/second.pytorch/second/pytorch/pillar_feature_rank_DAWN_catten_before/" + \
            #                   "%06d.npy" % example_index[spatial_features_index], output_pillar_feature)
                    
            #         # output fg bg feature 
            #         bg_select = torch.randperm(background_spatial_features.shape[0])[:check_foreground_num*3]
            #         background_spatial_features = background_spatial_features[bg_select].cpu().detach().numpy() 
            #         background_grid_distance = background_grid_distance[bg_select].cpu().detach().numpy()[:,np.newaxis]
            #         label_fg_bg = np.concatenate((np.ones((check_foreground_num,1)),np.zeros((background_spatial_features.shape[0],1))), axis=0)    
            #         d_fg_bg = np.concatenate((foreground_grid_distance,background_grid_distance), axis=0)
                    
            #         output_pillar_feature = np.concatenate((foreground_spatial_features, background_spatial_features), axis=0)
            #         output_pillar_feature = np.concatenate((output_pillar_feature, label_fg_bg, d_fg_bg), axis=1)
                    
            #         np.save("/data2/chihjen/second.pytorch/second/pytorch/pillar_feature_rank_DAWN_catten_before_fgbg/" + \
            #                   "%06d.npy" % example_index[spatial_features_index], output_pillar_feature)
                        
            
                        
            # example_index = example["image_idx"]
            # for spatial_features_index in range(feature_after.shape[0]):
            #     # for car
            #     foreground_grid_num = \
            #         np.load("/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object/training/label_grid_num/" + \
            #                   "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)
                        
            #     # for people
            #     # foreground_grid_num = \
            #     #     np.load("/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object/training/label_grid_num_person/" + \
            #     #               "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)
                
            #     # grid distance
            #     _, grid_distance = np.mgrid[0:foreground_grid_num.shape[1], 0:foreground_grid_num.shape[2]]
            #     grid_distance = foreground_grid_num.shape[2] - grid_distance
                        
            #     foreground_grid_flag = torch.from_numpy(np.squeeze(foreground_grid_num) > 0).cuda()
            #     foreground_grid_num_column = torch.from_numpy(np.squeeze(foreground_grid_num))[foreground_grid_flag].cuda()
            #     foreground_grid_distance = torch.from_numpy(grid_distance)[foreground_grid_flag].cuda()
            #     background_grid_distance = torch.from_numpy(grid_distance)[~foreground_grid_flag].cuda()
                
            #     foreground_spatial_features = feature_after[spatial_features_index,:,:,:].permute(1,2,0)[foreground_grid_flag].cuda()
            #     background_spatial_features = feature_after[spatial_features_index,:,:,:].permute(1,2,0)[~foreground_grid_flag].cuda()
                
            #     check_foreground_num = foreground_grid_num_column.shape[0]
            #     if check_foreground_num == 0:
            #         pass
            #     else:
            #         foreground_grid_num_column = np.reshape(foreground_grid_num_column.cpu().detach().numpy(),(check_foreground_num,1))
            #         foreground_spatial_features = foreground_spatial_features.cpu().detach().numpy()
            #         foreground_grid_distance = np.reshape(foreground_grid_distance.cpu().detach().numpy(),(check_foreground_num,1))
            #         output_pillar_feature = np.concatenate((foreground_spatial_features, foreground_grid_num_column, foreground_grid_distance), axis=1)
                    
            #         # # output bg feature 
            #         # bg_select = torch.randperm(background_spatial_features.shape[0])[:check_foreground_num*3]
            #         # background_spatial_features = background_spatial_features[bg_select].cpu().detach().numpy() 
            #         # background_grid_distance = background_grid_distance[bg_select].cpu().detach().numpy()[:,np.newaxis]
            #         # label_fg_bg = np.concatenate((np.ones((check_foreground_num,1)),np.zeros((background_spatial_features.shape[0],1))), axis=0)    
            #         # d_fg_bg = np.concatenate((foreground_grid_distance,background_grid_distance), axis=0)
                    
            #         # output_pillar_feature = np.concatenate((foreground_spatial_features, background_spatial_features), axis=0)
            #         # output_pillar_feature = np.concatenate((output_pillar_feature, label_fg_bg, d_fg_bg), axis=1)
                    
            #         np.save("/data2/chihjen/second.pytorch/second/pytorch/pillar_feature_after_atten_noRANK/" + \
            #                   "%06d.npy" % example_index[spatial_features_index], output_pillar_feature)
                        
            # for spatial_features_index in range(feature_before.shape[0]):                
            #     # for car
            #     foreground_grid_num = \
            #         np.load("/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object/training/label_grid_num/" + \
            #                   "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)                        
            #     # for people
            #     # foreground_grid_num = \
            #     #     np.load("/data2/chihjen/Kitti_dataset/PointRCNN/data/KITTI/object/training/label_grid_num_person/" + \
            #     #               "%06d.npy" % example_index[spatial_features_index]).transpose(2,1,0)
                
            #     # grid distance
            #     _, grid_distance = np.mgrid[0:foreground_grid_num.shape[1], 0:foreground_grid_num.shape[2]]
            #     grid_distance = foreground_grid_num.shape[2] - grid_distance
                        
            #     foreground_grid_flag = torch.from_numpy(np.squeeze(foreground_grid_num) > 0).cuda()
            #     foreground_grid_num_column = torch.from_numpy(np.squeeze(foreground_grid_num))[foreground_grid_flag].cuda()
            #     foreground_grid_distance = torch.from_numpy(grid_distance)[foreground_grid_flag].cuda()
            #     background_grid_distance = torch.from_numpy(grid_distance)[~foreground_grid_flag].cuda()
                
            #     foreground_spatial_features = feature_before[spatial_features_index,:,:,:].permute(1,2,0)[foreground_grid_flag].cuda()
            #     background_spatial_features = feature_before[spatial_features_index,:,:,:].permute(1,2,0)[~foreground_grid_flag].cuda()
                
            #     check_foreground_num = foreground_grid_num_column.shape[0]
            #     if check_foreground_num == 0:
            #         pass
            #     else:
            #         foreground_grid_num_column = np.reshape(foreground_grid_num_column.cpu().detach().numpy(),(check_foreground_num,1))
            #         foreground_spatial_features = foreground_spatial_features.cpu().detach().numpy()
            #         foreground_grid_distance = np.reshape(foreground_grid_distance.cpu().detach().numpy(),(check_foreground_num,1))
            #         output_pillar_feature = np.concatenate((foreground_spatial_features, foreground_grid_num_column, foreground_grid_distance), axis=1)
                    
            #         # # output bg feature 
            #         # bg_select = torch.randperm(background_spatial_features.shape[0])[:check_foreground_num*3]
            #         # background_spatial_features = background_spatial_features[bg_select].cpu().detach().numpy() 
            #         # background_grid_distance = background_grid_distance[bg_select].cpu().detach().numpy()[:,np.newaxis]
            #         # label_fg_bg = np.concatenate((np.ones((check_foreground_num,1)),np.zeros((background_spatial_features.shape[0],1))), axis=0)    
            #         # d_fg_bg = np.concatenate((foreground_grid_distance,background_grid_distance), axis=0)
                    
            #         # output_pillar_feature = np.concatenate((foreground_spatial_features, background_spatial_features), axis=0)
            #         # output_pillar_feature = np.concatenate((output_pillar_feature, label_fg_bg, d_fg_bg), axis=1)
                    
            #         np.save("/data2/chihjen/second.pytorch/second/pytorch/pillar_feature_before_atten_noRANK/" + \
            #                   "%06d.npy" % example_index[spatial_features_index], output_pillar_feature)
                
            ##################################  pillar feature output after 2d extract  #########################################

            return self.predict(example, preds_dict)

    def predict(self, example, preds_dict):
        t = time.time()
        batch_size = example['anchors'].shape[0]
        batch_anchors = example["anchors"].view(batch_size, -1, 7)

        # np.save("/data2/chihjen/second.pytorch/second/123.npy", batch_anchors.cpu().numpy())

        self._total_inference_count += batch_size
        batch_rect = example["rect"]
        batch_Trv2c = example["Trv2c"]
        batch_P2 = example["P2"]
        # anchors_mask 是用來遮住raw point cloud原本沒點的地方
        if "anchors_mask" not in example:
            batch_anchors_mask = [None] * batch_size
        else:
            batch_anchors_mask = example["anchors_mask"].view(batch_size, -1)
            # print(batch_anchors_mask)
        batch_imgidx = example['image_idx']


        self._total_forward_time += time.time() - t
        t = time.time()
        batch_box_preds = preds_dict["box_preds"]
        batch_cls_preds = preds_dict["cls_preds"]
        

        batch_box_preds = batch_box_preds.view(batch_size, -1,
                                               self._box_coder.code_size)

        num_class_with_bg = self._num_class
        if not self._encode_background_as_zeros:
            num_class_with_bg = self._num_class + 1
            
        # np.save("/data2/chihjen/second.pytorch/second/batch_anchors.npy", batch_anchors.detach().cpu().numpy())
        # np.save("/data2/chihjen/second.pytorch/second/batch_box_preds_before.npy", batch_box_preds.detach().cpu().numpy())

        batch_cls_preds = batch_cls_preds.view(batch_size, -1,
                                               num_class_with_bg)

        batch_box_preds = self._box_coder.decode_torch(batch_box_preds,
                                                       batch_anchors)
        
        # np.save("/data2/chihjen/second.pytorch/second/batch_box_preds_after.npy", batch_box_preds.detach().cpu().numpy())
        
        if self._use_direction_classifier:
            batch_dir_preds = preds_dict["dir_cls_preds"]
            batch_dir_preds = batch_dir_preds.view(batch_size, -1, 2)
        else:
            batch_dir_preds = [None] * batch_size

        predictions_dicts = []
        for box_preds, cls_preds, dir_preds, rect, Trv2c, P2, img_idx, a_mask in zip(
                batch_box_preds, batch_cls_preds, batch_dir_preds, batch_rect,
                batch_Trv2c, batch_P2, batch_imgidx, batch_anchors_mask
        ):
            
            if a_mask is not None:
                box_preds = box_preds[a_mask]
                # print(box_preds.shape)
                cls_preds_o = cls_preds*1.0
                cls_preds = cls_preds[a_mask]
            if self._use_direction_classifier:
                if a_mask is not None:
                    dir_preds = dir_preds[a_mask]
                # print(dir_preds.shape)
                dir_labels = torch.max(dir_preds, dim=-1)[1]
            if self._encode_background_as_zeros:
                # this don't support softmax
                assert self._use_sigmoid_score is True
                total_scores = torch.sigmoid(cls_preds)
                cls_preds_o = torch.sigmoid(cls_preds_o)
            else:
                # encode background as first element in one-hot vector
                if self._use_sigmoid_score:
                    total_scores = torch.sigmoid(cls_preds)[..., 1:]
                    cls_preds_o = torch.sigmoid(cls_preds_o)
                else:
                    total_scores = F.softmax(cls_preds, dim=-1)[..., 1:]
                    cls_preds_o = torch.sigmoid(cls_preds_o)
            # Apply NMS in birdeye view
            if self._use_rotate_nms:
                nms_func = box_torch_ops.rotate_nms
            else:
                nms_func = box_torch_ops.nms
            selected_boxes = None
            selected_labels = None
            selected_scores = None
            selected_dir_labels = None

            if self._multiclass_nms:
                # curently only support class-agnostic boxes.
                boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                if not self._use_rotate_nms:
                    box_preds_corners = box_torch_ops.center_to_corner_box2d(
                        boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                        boxes_for_nms[:, 4])
                    boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                        box_preds_corners)
                boxes_for_mcnms = boxes_for_nms.unsqueeze(1)
                selected_per_class = box_torch_ops.multiclass_nms(
                    nms_func=nms_func,
                    boxes=boxes_for_mcnms,
                    scores=total_scores,
                    num_class=self._num_class,
                    pre_max_size=self._nms_pre_max_size,
                    post_max_size=self._nms_post_max_size,
                    iou_threshold=self._nms_iou_threshold,
                    score_thresh=self._nms_score_threshold,
                )
                selected_boxes, selected_labels, selected_scores = [], [], []
                selected_dir_labels = []
                for i, selected in enumerate(selected_per_class):
                    if selected is not None:
                        num_dets = selected.shape[0]
                        selected_boxes.append(box_preds[selected])
                        selected_labels.append(
                            torch.full([num_dets], i, dtype=torch.int64))
                        if self._use_direction_classifier:
                            selected_dir_labels.append(dir_labels[selected])
                        selected_scores.append(total_scores[selected, i])
                if len(selected_boxes) > 0:
                    selected_boxes = torch.cat(selected_boxes, dim=0)
                    selected_labels = torch.cat(selected_labels, dim=0)
                    selected_scores = torch.cat(selected_scores, dim=0)
                    if self._use_direction_classifier:
                        selected_dir_labels = torch.cat(
                            selected_dir_labels, dim=0)
                else:
                    selected_boxes = None
                    selected_labels = None
                    selected_scores = None
                    selected_dir_labels = None
            else:
                # get highest score per prediction, than apply nms
                # to remove overlapped box.
                if num_class_with_bg == 1:
                    top_scores = total_scores.squeeze(-1)
                    top_labels = torch.zeros(
                        total_scores.shape[0],
                        device=total_scores.device,
                        dtype=torch.long)
                else:
                    top_scores, top_labels = torch.max(total_scores, dim=-1)
                if self._nms_score_threshold > 0.0:
                    thresh = torch.tensor(
                        [self._nms_score_threshold],
                        device=total_scores.device).type_as(total_scores)
                    top_scores_keep = (top_scores >= thresh)
############################################################################################################################################  
                    # # print(cls_preds_o.shape)
                    # # print(foreground_flag.shape)
                    # # print("\n")
                    # out_cls_acc = torch.cat([foreground_flag, cls_preds_o.detach(), grid_distance], 1)
                    # np.save("/data2/chihjen/second.pytorch/second/cls_acc_DAWN_RANK/" + "%06d.npy" % img_idx, out_cls_acc.cpu().numpy())
############################################################################################################################################
                    top_scores = top_scores.masked_select(top_scores_keep)
                if top_scores.shape[0] != 0:
                    if self._nms_score_threshold > 0.0:
                        box_preds = box_preds[top_scores_keep]
                        if self._use_direction_classifier:
                            dir_labels = dir_labels[top_scores_keep]
                        top_labels = top_labels[top_scores_keep]
                    boxes_for_nms = box_preds[:, [0, 1, 3, 4, 6]]
                    if not self._use_rotate_nms:
                        box_preds_corners = box_torch_ops.center_to_corner_box2d(
                            boxes_for_nms[:, :2], boxes_for_nms[:, 2:4],
                            boxes_for_nms[:, 4])
                        boxes_for_nms = box_torch_ops.corner_to_standup_nd(
                            box_preds_corners)
                    # the nms in 3d detection just remove overlap boxes.
                    selected = nms_func(
                        boxes_for_nms,
                        top_scores,
                        pre_max_size=self._nms_pre_max_size,
                        post_max_size=self._nms_post_max_size,
                        iou_threshold=self._nms_iou_threshold,
                    )
                else:
                    selected = None
                if selected is not None:
                    selected_boxes = box_preds[selected]
                    # print(selected_boxes[:,0])
                    if self._use_direction_classifier:
                        selected_dir_labels = dir_labels[selected]
                    selected_labels = top_labels[selected]
                    selected_scores = top_scores[selected]
            # finally generate predictions.

            if selected_boxes is not None:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                if self._use_direction_classifier:
                    dir_labels = selected_dir_labels
                    # opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.byte()
                    opp_labels = (box_preds[..., -1] > 0) ^ dir_labels.to(torch.bool)
                    box_preds[..., -1] += torch.where(
                        opp_labels,
                        torch.tensor(np.pi).type_as(box_preds),
                        torch.tensor(0.0).type_as(box_preds))
                    # box_preds[..., -1] += (
                    #     ~(dir_labels.byte())).type_as(box_preds) * np.pi
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                final_box_preds_camera = box_torch_ops.box_lidar_to_camera(
                    final_box_preds, rect, Trv2c)
                locs = final_box_preds_camera[:, :3]
                dims = final_box_preds_camera[:, 3:6]
                angles = final_box_preds_camera[:, 6]
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_torch_ops.center_to_corner_box3d(
                    locs, dims, angles, camera_box_origin, axis=1)
                box_corners_in_image = box_torch_ops.project_to_image(
                    box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = torch.min(box_corners_in_image, dim=1)[0]
                maxxy = torch.max(box_corners_in_image, dim=1)[0]
                # minx = torch.min(box_corners_in_image[..., 0], dim=1)[0]
                # maxx = torch.max(box_corners_in_image[..., 0], dim=1)[0]
                # miny = torch.min(box_corners_in_image[..., 1], dim=1)[0]
                # maxy = torch.max(box_corners_in_image[..., 1], dim=1)[0]
                # box_2d_preds = torch.stack([minx, miny, maxx, maxy], dim=1)
                box_2d_preds = torch.cat([minxy, maxxy], dim=1)
                # predictions
                predictions_dict = {
                    "bbox": box_2d_preds,
                    "box3d_camera": final_box_preds_camera,
                    "box3d_lidar": final_box_preds,
                    "scores": final_scores,
                    "label_preds": label_preds,
                    "image_idx": img_idx,
                }
            else:
                predictions_dict = {
                    "bbox": None,
                    "box3d_camera": None,
                    "box3d_lidar": None,
                    "scores": None,
                    "label_preds": None,
                    "image_idx": img_idx,
                }
            predictions_dicts.append(predictions_dict)
        
        self._total_postprocess_time += time.time() - t
        return predictions_dicts

    @property
    def avg_forward_time(self):
        return self._total_forward_time / self._total_inference_count

    @property
    def avg_postprocess_time(self):
        return self._total_postprocess_time / self._total_inference_count

    def clear_time_metrics(self):
        self._total_forward_time = 0.0
        self._total_postprocess_time = 0.0
        self._total_inference_count = 0

    def metrics_to_float(self):
        self.rpn_acc.float()
        self.rpn_metrics.float()
        self.rpn_cls_loss.float()
        self.rpn_loc_loss.float()
        self.rpn_total_loss.float()

    def update_metrics(self,
                       cls_loss,
                       loc_loss,
                       cls_preds,
                       labels,
                       sampled):
        batch_size = cls_preds.shape[0]
        num_class = self._num_class
        if not self._encode_background_as_zeros:
            num_class += 1
        cls_preds = cls_preds.view(batch_size, -1, num_class)
        rpn_acc = self.rpn_acc(labels, cls_preds, sampled).numpy()[0]
        prec, recall = self.rpn_metrics(labels, cls_preds, sampled)
        prec = prec.numpy()
        recall = recall.numpy()
        rpn_cls_loss = self.rpn_cls_loss(cls_loss).numpy()[0]
        rpn_loc_loss = self.rpn_loc_loss(loc_loss).numpy()[0]
        ret = {
            "cls_loss": float(rpn_cls_loss),
            "cls_loss_rt": float(cls_loss.data.cpu().numpy()),
            'loc_loss': float(rpn_loc_loss),
            "loc_loss_rt": float(loc_loss.data.cpu().numpy()),
            "rpn_acc": float(rpn_acc),
        }
        for i, thresh in enumerate(self.rpn_metrics.thresholds):
            ret[f"prec@{int(thresh*100)}"] = float(prec[i])
            ret[f"rec@{int(thresh*100)}"] = float(recall[i])
        return ret

    def clear_metrics(self):
        self.rpn_acc.clear()
        self.rpn_metrics.clear()
        self.rpn_cls_loss.clear()
        self.rpn_loc_loss.clear()
        self.rpn_total_loss.clear()

    @staticmethod
    def convert_norm_to_float(net):
        '''
        BatchNorm layers to have parameters in single precision.
        Find all layers and convert them back to float. This can't
        be done with built in .apply as that function will apply
        fn to all modules, parameters, and buffers. Thus we wouldn't
        be able to guard the float conversion based on the module type.
        '''
        if isinstance(net, torch.nn.modules.batchnorm._BatchNorm):
            net.float()
        for child in net.children():
            VoxelNet.convert_norm_to_float(net)
        return net


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(
        boxes2[..., -1:])
    rad_tg_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])
    boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    boxes2 = torch.cat([boxes2[..., :-1], rad_tg_encoding], dim=-1)
    return boxes1, boxes2


def create_loss(loc_loss_ftor,
                cls_loss_ftor,
                box_preds,
                cls_preds,
                cls_targets,
                cls_weights,
                reg_targets,
                reg_weights,
                num_class,
                encode_background_as_zeros=True,
                encode_rad_error_by_sin=True,
                box_code_size=7):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)
    if encode_background_as_zeros:
        cls_preds = cls_preds.view(batch_size, -1, num_class)
    else:
        cls_preds = cls_preds.view(batch_size, -1, num_class + 1)
    cls_targets = cls_targets.squeeze(-1)
    one_hot_targets = torchplus.nn.one_hot(
        cls_targets, depth=num_class + 1, dtype=box_preds.dtype)
    if encode_background_as_zeros:
        one_hot_targets = one_hot_targets[..., 1:]
    if encode_rad_error_by_sin:
        # sin(a - b) = sinacosb-cosasinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)
    loc_losses = loc_loss_ftor(
        box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(
        cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
    return loc_losses, cls_losses


def prepare_loss_weights(labels,
                         pos_cls_weight=1.0,
                         neg_cls_weight=1.0,
                         loss_norm_type=LossNormType.NormByNumPositives,
                         dtype=torch.float32):
    """get cls_weights and reg_weights from labels.
    """
    cared = labels >= 0
    # cared: [N, num_anchors]
    positives = labels > 0
    negatives = labels == 0
    negative_cls_weights = negatives.type(dtype) * neg_cls_weight
    cls_weights = negative_cls_weights + pos_cls_weight * positives.type(dtype)
    reg_weights = positives.type(dtype)
    if loss_norm_type == LossNormType.NormByNumExamples:
        num_examples = cared.type(dtype).sum(1, keepdim=True)
        num_examples = torch.clamp(num_examples, min=1.0)
        cls_weights /= num_examples
        bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(bbox_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPositives:  # for focal loss
        pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    elif loss_norm_type == LossNormType.NormByNumPosNeg:
        pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
        normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
        cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
        cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
        # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
        normalizer = torch.clamp(normalizer, min=1.0)
        reg_weights /= normalizer[:, 0:1, 0]
        cls_weights /= cls_normalizer
    else:
        raise ValueError(
            f"unknown loss norm type. available: {list(LossNormType)}")
    return cls_weights, reg_weights, cared


def assign_weight_to_each_class(labels,
                                weight_per_class,
                                norm_by_num=True,
                                dtype=torch.float32):
    weights = torch.zeros(labels.shape, dtype=dtype, device=labels.device)
    for label, weight in weight_per_class:
        positives = (labels == label).type(dtype)
        weight_class = weight * positives
        if norm_by_num:
            normalizer = positives.sum()
            normalizer = torch.clamp(normalizer, min=1.0)
            weight_class /= normalizer
        weights += weight_class
    return weights


def get_direction_target(anchors, reg_targets, one_hot=True):
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, 7)
    rot_gt = reg_targets[..., -1] + anchors[..., -1]
    dir_cls_targets = (rot_gt > 0).long()
    if one_hot:
        dir_cls_targets = torchplus.nn.one_hot(
            dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets
