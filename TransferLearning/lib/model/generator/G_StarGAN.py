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

class Generator(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, c_dim=5, repeat_num=6):
        super(Generator, self).__init__()

        layers = []
        # layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Conv2d(64, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=1, bias=False))
            # layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            # layers.append(nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=3, stride=2, padding=1, bias=False))
            # layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=True))
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim, affine=True, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
            # curr_dim = curr_dim // 2

        # layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Conv2d(curr_dim, 512, kernel_size=3, stride=1, padding=1, bias=False))
        # layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x, y):
        # Replicate spatially and concatenate domain information.
        # c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, x.size(2), x.size(3))
        # x = torch.cat([x, c], dim=1)
        avgp = nn.AdaptiveAvgPool2d((y.size(2), y.size(3)))
        return avgp(self.main(x))

class Generator_ins(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=512, c_dim=5, repeat_num=2):
        super(Generator_ins, self).__init__()

        layers = []
        # layers.append(nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.Conv2d(512, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        # layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        # layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim // 2, kernel_size=3, stride=1, padding=1, bias=False))
            # layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.InstanceNorm2d(curr_dim // 2, affine=True, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=1, padding=1, bias=False))
            # layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=True))
            layers.append(nn.InstanceNorm2d(curr_dim * 2, affine=True, track_running_stats=False))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2


        # layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        # layers.append(nn.Conv2d(curr_dim, 512, kernel_size=3, stride=1, padding=1, bias=False))
        # layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        # Replicate spatially and concatenate domain information.
        # c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, x.size(2), x.size(3))
        # x = torch.cat([x, c], dim=1)
        # avgp = nn.AdaptiveAvgPool2d((y.size(2), y.size(3)))
        return self.main(x)

class Generator_img_res(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, repeat_num=3):
        super(Generator_img_res, self).__init__()

        layers = []
        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
            layers.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
            layers.append(Bottleneck(dim_in=conv_dim, dim_out= 2*conv_dim))
            conv_dim = conv_dim * 2
        self.main = nn.Sequential(*layers)
    def forward(self, x, y):
        # Replicate spatially and concatenate domain information.
        # c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, x.size(2), x.size(3))
        # x = torch.cat([x, c], dim=1)
        avgp = nn.AdaptiveAvgPool2d((y.size(2), y.size(3)))
        return avgp(self.main(x))

class Generator_img_res_cat(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, repeat_num=3):
        super(Generator_img_res_cat, self).__init__()

        layers = []
        # Bottleneck layers.
        layers.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers.append(Bottleneck(dim_in=conv_dim, dim_out=conv_dim))

        for i in range(1, repeat_num):
            layers.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
            layers.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
            layers.append(Bottleneck(dim_in=conv_dim, dim_out= 2*conv_dim))
            conv_dim = conv_dim * 2
        self.main = nn.Sequential(*layers)

    def forward(self, x, y):
        # Replicate spatially and concatenate domain information.
        # c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, x.size(2), x.size(3))
        # x = torch.cat([x, c], dim=1)
        avgp = nn.AdaptiveAvgPool2d((y.size(2), y.size(3)))
        return avgp(self.main(x))

class Generator_img_res6(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, repeat_num=3):
        super(Generator_img_res6, self).__init__()

        layers1 = []
        layers1.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers1.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers1.append(Bottleneck(dim_in=conv_dim, dim_out= 2*conv_dim))
        conv_dim = conv_dim * 2

        layers2 = []
        layers2.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers2.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers2.append(Bottleneck(dim_in=conv_dim, dim_out=2 * conv_dim))
        conv_dim = conv_dim * 2

        layers3 = []
        layers3.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers3.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers3.append(Bottleneck(dim_in=conv_dim, dim_out=2 * conv_dim))
        conv_dim = conv_dim * 2

        layers4=[]
        layers4.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
        layers4.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))

        self.layers1 = nn.Sequential(*layers1)
        self.layers2 = nn.Sequential(*layers2)
        self.layers3 = nn.Sequential(*layers3)
        self.layers4 = nn.Sequential(*layers4)
        self.glb = nn.AdaptiveAvgPool2d((1,1))

        cab1 = []
        cab1.append(nn.Conv2d(512+64+128, 512, kernel_size=1, bias=False))
        cab1.append(nn.ReLU(inplace=True))
        cab1.append(nn.Conv2d(512, 128, kernel_size=1, bias=False))
        cab1.append(nn.Sigmoid())

        cab2 = []
        cab2.append(nn.Conv2d(512 + 64 + 256, 512, kernel_size=1, bias=False))
        cab2.append(nn.ReLU(inplace=True))
        cab2.append(nn.Conv2d(512, 256, kernel_size=1, bias=False))
        cab2.append(nn.Sigmoid())

        cab3 = []
        cab3.append(nn.Conv2d(512 + 64 + 512, 512, kernel_size=1, bias=False))
        cab3.append(nn.ReLU(inplace=True))
        cab3.append(nn.Conv2d(512, 512, kernel_size=1, bias=False))
        cab3.append(nn.Sigmoid())

        self.cab1 = nn.Sequential(*cab1)
        self.cab2 = nn.Sequential(*cab2)
        self.cab3 = nn.Sequential(*cab3)

    def forward(self, x, S,T):
        # Replicate spatially and concatenate domain information.
        # c = c.view(c.size(0), c.size(1), 1, 1)
        # c = c.repeat(1, 1, x.size(2), x.size(3))
        # x = torch.cat([x, c], dim=1)
        avgp = nn.AdaptiveAvgPool2d((S.size(2), S.size(3)))

        x = self.layers1(x)
        att1 = self.cab1(torch.cat((self.glb(x),self.glb(S), self.glb(T)), 1))
        x = x + att1 * x

        x = self.layers2(x)
        att2 = self.cab2(torch.cat((self.glb(x), self.glb(S), self.glb(T)), 1))
        x = x + att2 * x

        x = self.layers3(x)
        att3 = self.cab3(torch.cat((self.glb(x), self.glb(S), self.glb(T)), 1))
        x = x + att3 * x

        x = self.layers4(x)

        return avgp(x)


# class Generator_ins_res(nn.Module):
#     """Generator network."""
#
#     def __init__(self, conv_dim=64, repeat_num=2):
#         super(Generator_ins_res, self).__init__()
#
#         layers = []
#         # Bottleneck layers.
#         for i in range(repeat_num):
#             layers.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
#             # layers.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
#             layers.append(Bottleneck(dim_in=conv_dim, dim_out= 2*conv_dim))
#             conv_dim = conv_dim * 2
#         layers.append(Bottleneck_NoPool(dim_in=conv_dim, dim_out=2*conv_dim))
#         self.main = nn.Sequential(*layers)
#         # self.main.layer.4[].weight.data.normal_(mean, stddev)
#         for key, value in dict(self.main.named_parameters()).items():
#             if 'bias' in key:
#                 value.data.zero_()
#             else:
#                 value.data.normal_(0, 0.01)
#
#     def forward(self, x):
#         return self.main(x)

class Generator_ins_res(nn.Module):
    """Generator network."""

    def __init__(self, conv_dim=64, repeat_num=3):
        super(Generator_ins_res, self).__init__()

        layers = []
        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
            layers.append(ResidualBlock(dim_in=conv_dim, dim_out=conv_dim))
            layers.append(Bottleneck(dim_in=conv_dim, dim_out= 2*conv_dim))
            conv_dim = conv_dim * 2
        self.main = nn.Sequential(*layers)
    def forward(self, x):
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