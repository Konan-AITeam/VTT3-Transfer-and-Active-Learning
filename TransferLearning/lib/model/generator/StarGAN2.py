import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import cv2

class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(dim_in, dim_out, bias=False),
            # nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out, bias=False),
            # nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))
        )
    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=2):
        super(Generator, self).__init__()

        layers = []
        # layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Linear(4096, 2048, bias=False))
        # layers.append(nn.InstanceNorm2d(2048, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = 2048
        for i in range(2):
            layers.append(nn.Linear(curr_dim, curr_dim // 2,bias=False))
            # layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.Linear(curr_dim, curr_dim * 2, bias=False))
            # layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Linear(curr_dim, 4096, bias=False))
        # layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, x.size(2), x.size(3))
        # x = torch.cat([x, c], dim=1)
        return self.main(x)


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, image_size=128, conv_dim=64, c_dim=5, repeat_num=4):
        super(Discriminator, self).__init__()
        layers = []
        # layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.Conv2d(128, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 2, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)

    def forward(self, x, label):
        h = self.main(x)
        out_src = self.conv1(h)
        label = LabelResizeLayer_im(out_src, label)
        loss = F.cross_entropy(out_src, label)

        return loss



def LabelResizeLayer_im(feats, lbs):
    lbs = lbs.data.cpu().numpy()
    lbs_resize = cv2.resize(lbs, (feats.shape[3], feats.shape[2]), interpolation=cv2.INTER_NEAREST)

    gt_blob = np.zeros((1, lbs_resize.shape[0], lbs_resize.shape[1], 1), dtype=np.float32)
    gt_blob[0, 0:lbs_resize.shape[0], 0:lbs_resize.shape[1], 0] = lbs_resize

    channel_swap = (0, 3, 1, 2)
    gt_blob = gt_blob.transpose(channel_swap).astype(int)

    # gt_blob_onehot = np.zeros((gt_blob.shape[0], 2, gt_blob.shape[2], gt_blob.shape[3]))
    #
    # gt_blob_onehot[0, gt_blob[0,:,:]] = 1

    gt_blob = torch.squeeze(Variable(torch.from_numpy(gt_blob).long().cuda(), requires_grad=False), dim=1)
    return gt_blob