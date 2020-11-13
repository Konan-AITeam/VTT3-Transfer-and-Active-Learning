import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import cv2
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.utils.config import cfg

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class Bottleneck(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(Bottleneck, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=2, bias=False),
            # nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

        self.main2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=2, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return self.main2(x) + self.main(x)


class Bottleneck_NoPool(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(Bottleneck_NoPool, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
            # nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

        self.main2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return self.main2(x) + self.main(x)

class RoI_pooling(nn.Module):
    def __init__(self):
        super(RoI_pooling, self).__init__()
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

    def forward(self, base_feat, rois):
        pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        return pooled_feat

class Generator_IAM(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, repeat_num=3):
        super(Generator_IAM, self).__init__()

        layers1 = []
        layers1.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers1.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers1.append(Bottleneck(dim_in=conv_dim, dim_out=2 * conv_dim))
        conv_dim = conv_dim * 2
        self.layer1=nn.Sequential(*layers1)
        self.IAM_att_1 = nn.Conv2d(conv_dim, conv_dim//4, 1, bias=False)
        self.IAM_att_2 = nn.Conv2d(conv_dim//4, conv_dim, 1, bias=False)
        self.IAM_ft_1 = nn.Conv2d(conv_dim, conv_dim//4, 1, bias=False)
        self.IAM_ft_2 = nn.Conv2d(conv_dim//4, conv_dim//4, 3, bias=False)
        self.IAM_ft_3 = nn.Conv2d(conv_dim // 4, conv_dim // 4, 3, bias=False)


        layers2 = []
        layers2.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers2.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers2.append(Bottleneck(dim_in=conv_dim, dim_out=2 * conv_dim))
        conv_dim = conv_dim * 2
        self.layer2 = nn.Sequential(*layers1)

        layers3 = []
        layers3.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers3.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers3.append(Bottleneck(dim_in=conv_dim, dim_out=2 * conv_dim))
        conv_dim = conv_dim * 2
        self.layer3 = nn.Sequential(*layers1)

        self.RCNN_roi_align1 = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 4.0)
        self.RCNN_roi_align2 = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 8.0)
        self.RCNN_roi_align3 = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)


        # for i in range(repeat_num):
        #     layers.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        #     layers.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        #     layers.append(Bottleneck(dim_in=conv_dim, dim_out= 2*conv_dim))
        #     conv_dim = conv_dim * 2
        # self.main = nn.Sequential(*layers)
        self.roip = RoI_pooling()
    def forward(self, x, y, rois):
        avgp = nn.AdaptiveAvgPool2d((y.size(2), y.size(3)))
        return avgp(self.main(x))



