import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # print('backward')
        grad_input = - 0.1 * grad_output.clone()
        return grad_input


# class GradZero(nn.Module):
#     def forward(self, x):
#         return x
#     def backward(self, grad_output):
#         return (0 * grad_output)

# class ImageLevelDA_NoReverse(nn.Module):
#     def __init__(self):
#         super(ImageLevelDA_NoReverse, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(512, 512, 1),
#             nn.ReLU(),
#             nn.Conv2d(512, 2, 1)
#         )
#
#     def forward(self, feat, label):
#         # feat = GradReverse.apply(feat)
#         feat = self.layers(feat)
#
#         label = LabelResizeLayer_im(feat, label)
#         loss = F.cross_entropy(feat, label)
#         return loss

class ImageLevelDA(nn.Module):
    def __init__(self):
        super(ImageLevelDA, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(),
            nn.Conv2d(1024, 2, 1)
        )

    def forward(self, feat, label):
        feat = GradReverse.apply(feat)
        feat = self.layers(feat)

        label = LabelResizeLayer_im(feat, label)
        loss = F.cross_entropy(feat, label)
        return loss

# class ImageLevelDA_pgrl(nn.Module):
#     def __init__(self):
#         super(ImageLevelDA_pgrl, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(1024, 1024, 1),
#             nn.ReLU(),
#             nn.Conv2d(1024, 2, 1)
#         )
#
#     def forward(self, feat, label, need_backprop, pgrl_label):
#         if need_backprop.numpy():
#             feat = PartialGradReverse.apply(feat, pgrl_label)
#         else:
#             feat = GradReverse.apply(feat)
#         feat = self.layers(feat)
#
#         label = LabelResizeLayer_im(feat, label)
#         # print(label.size())
#         loss = F.cross_entropy(feat, label)
#         return loss

# class PartialGradReverse(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, pgrl_label):
#         ctx.save_for_backward(pgrl_label)
#         return x.clone()
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         # print('backward')
#         pgrl_label = ctx.saved_variables[0]
#         grad_input = - 0.1 * pgrl_label * grad_output.clone()
#         return grad_input, 0 * pgrl_label


# class GradZero(nn.Module):
#     def forward(self, x):
#         return x
#     def backward(self, grad_output):
#         return (0 * grad_output)

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

class InstanceLevelDA(nn.Module):
    def __init__(self):
        super(InstanceLevelDA, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1)
        )
        # self.resize = LabelResizeLayer_in()

    def forward(self, feat, label):
        # feat = GradReverse.apply(feat)
        # print(feat.shape)
        feat = self.layers(feat)
        label = LabelResizeLayer_in(feat, label)
        loss = F.binary_cross_entropy_with_logits(feat, label)

        return loss



def LabelResizeLayer_in(feats, lbs):
    # print(feats.shape)
    # print(lbs.shape)
    resized_lbs = np.ones((feats.shape[0], 1))
    resized_lbs[:] = lbs[:feats.shape[0], 0:1]
    resized_lbs = Variable(torch.from_numpy(resized_lbs).float().cuda(), requires_grad=False)
    return resized_lbs
