import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.utils.loss import grad_reverse
from model.rpn.rpn_origin import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
import cv2
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

from model.rpn.anchor_target_layer import _AnchorTargetLayer
from model.pgrl.target_anchor_search import _TargetAnchorSearch
from model.generator.G_StarGAN import Generator, Generator_ins, Discriminator, Generator_img_res, Generator_ins_res

class _da_fasterRCNN(nn.Module):
    """ domain adaptive faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_da_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        self.D_img = ImageLevelDA()

    def forward(self, im_data, im_info, gt_boxes, num_boxes, need_backprop=None, need_G_img=None, need_G_ins=None, dc_label=None):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        if self.training:
            DA_loss_img = 0.1 * self.D_img(GradReverse.apply(base_feat), dc_label)
        else:
            DA_loss_img = 0

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes, need_backprop)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            if need_backprop.numpy():
                roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
                rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
                # print(rois_label)
                rois_label = Variable(rois_label.view(-1).long())
                rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
                rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
                rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
            else:
                rois_label = None
                rois_target = None
                rois_inside_ws = None
                rois_outside_ws = None
                rpn_loss_cls = 0
                rpn_loss_bbox = 0
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)

        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))


        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            if need_backprop.numpy():
                # select the corresponding columns according to roi labels
                # print(bbox_pred.shape)
                bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
                bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
                bbox_pred = bbox_pred_select.squeeze(1) # gathers rois of the correspond class via rois_label

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        RCNN_loss_img = 0
        RCNN_loss_cst = 0

        if self.training:
            if need_backprop.numpy():
                # calculate classification and b.b. regression loss only for source data
                # classification loss
                RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

                # bounding box regression L1 loss
                RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
                ssda_loss = Variable(torch.zeros(1).float().cuda())

            else:
                x = grad_reverse(cls_score, 1.0)
                x = F.softmax(x)
                ssda_loss = 0.1 * torch.mean(torch.sum(x * (torch.log(x + 1e-5)), 1))

                RCNN_loss_cls = Variable(torch.zeros(1).float().cuda())
                RCNN_loss_bbox = Variable(torch.zeros(1).float().cuda())
                rpn_loss_cls = Variable(torch.zeros(1).float().cuda())
                rpn_loss_bbox = Variable(torch.zeros(1).float().cuda())

            # Domain Adaptation Components


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)


        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, DA_loss_img, ssda_loss#, DA_loss_ins

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        # normal_init(self.im_da.layers[0], 0, 0.001, cfg.TRAIN.TRUNCATED)
        # normal_init(self.im_da.layers[2], 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()


# class GradReverse(nn.Module):
#     def forward(self, x):
#         return x
#     def backward(self, grad_output):
#         return (-0.1*grad_output)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # print('backward')
        grad_input = - 0.1 * grad_output.clone()
        return grad_input

class GradReverseOne(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # print('backward')
        grad_input = - 0.25 * grad_output.clone()
        return grad_input

class GradReverseIns(torch.autograd.Function):
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


class ImageLevelDA(nn.Module):
    def __init__(self):
        super(ImageLevelDA, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(512, 512, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(512, 2, 1, bias=False)
        )
        self.layers[0].weight.data.normal_(0, 0.001)
        self.layers[2].weight.data.normal_(0, 0.001)

    def forward(self, feat, label):
        feat = self.layers(feat)
        label = LabelResizeLayer_im(feat, label)
        loss = F.cross_entropy(feat, label)
        return loss

class ImageLevelDA_ASPP(nn.Module):
    def __init__(self):
        super(ImageLevelDA_ASPP, self).__init__()
        self.aspp1 = nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.aspp2 = nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=3, dilation=3, bias=False)
        self.aspp3 = nn.Conv2d(512, 64, kernel_size=3, stride=2, padding=9, dilation=9, bias=False)

        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(192, 2, kernel_size=1, bias=False)
        )

    def forward(self, feat, label):
        # feat = GradReverse.apply(feat)
        feat = torch.cat((self.aspp1(feat), self.aspp2(feat), self.aspp3(feat)), 1)
        # feat = self.aspp1(feat)
        feat = self.layers(feat)

        label = LabelResizeLayer_im(feat, label)
        loss = F.cross_entropy(feat, label)
        return loss


class PartialGradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, pgrl_label):
        ctx.save_for_backward(pgrl_label)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # print('backward')
        pgrl_label = ctx.saved_variables[0]
        grad_input = - 0.1 * pgrl_label * grad_output.clone()
        return grad_input, 0 * pgrl_label


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
