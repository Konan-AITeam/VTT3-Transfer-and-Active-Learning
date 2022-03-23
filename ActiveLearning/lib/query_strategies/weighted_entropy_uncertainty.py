from cv2 import exp
import numpy as np
import time
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedEntropyUncertainty(object):
    def __init__(self, num_query):
        super(WeightedEntropyUncertainty, self).__init__()
        self.num_query = num_query
        self.entropy = WeightedEntropy()

    def query(self, data_loader, fasterRCNN):
        start = time.time()

        fasterRCNN.eval()
        queries_scores = []
        data_iter = iter(data_loader)
        iters_per_epoch = len(data_loader)
        for step in range(iters_per_epoch):
            # step_start = time.time()
            data = next(data_iter)
            im_data = Variable(data[0].cuda())
            im_info = Variable(data[1].cuda())
            # gt_boxes = Variable(data[2].cuda())
            # num_boxes = Variable(data[3].cuda())
            gt_boxes = Variable(torch.zeros([1, 4]).cuda())
            num_boxes = Variable(torch.zeros([1, 1]).cuda())
            im_paths = data[4][0]

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            score = self.entropy(cls_prob)

            queries_scores.append((im_paths, score))
            # step_duration = time.time() - step_start
            # print(step_duration)

        # sort
        queries_scores.sort(key=lambda element: element[1], reverse=True)
        queries_scores = queries_scores[:self.num_query]

        queries = [x[0] for x in queries_scores]
        scores = [x[1] for x in queries_scores]

        duration = time.time() - start
        print('query duration: %f' % duration)

        fasterRCNN.train()
        return queries, scores


class WeightedEntropyUncertainty_poolbased(object):
    def __init__(self, num_query):
        super(WeightedEntropyUncertainty_poolbased, self).__init__()
        self.num_query = num_query
        self.entropy = WeightedEntropy(temp=0.5)

    def query(self, data_loader, fasterRCNN, pool, epoch=None):
        print('before: %d' % len(pool))
        start = time.time()

        fasterRCNN.eval()
        queries_scores = []
        data_iter = iter(data_loader)
        iters_per_epoch = len(data_loader)
        for step in range(iters_per_epoch):
            data = next(data_iter)
            im_data = Variable(data[0].cuda())
            im_info = Variable(data[1].cuda())
            # gt_boxes = Variable(data[2].cuda())
            # num_boxes = Variable(data[3].cuda())
            gt_boxes = Variable(torch.zeros([1, 4]).cuda())
            num_boxes = Variable(torch.zeros([1, 1]).cuda())
            im_paths = data[4][0]

            if im_paths in pool:
                continue

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            score = self.entropy(cls_prob)

            queries_scores.append((im_paths, score))

            if step % 100 == 0:
                print('query step: %d' % step)

        # sort
        queries_scores.sort(key=lambda element: element[1], reverse=True) # collect high entropy
        # queries_scores.sort(key=lambda element: element[1]) # collect low entropy
        queries_scores = queries_scores[:self.num_query]

        queries = [x[0] for x in queries_scores]
        scores = [x[1] for x in queries_scores]

        pool += queries
        print('after: %d' % len(pool))

        duration = time.time() - start
        print('query duration: %f' % duration)

        # save queries
        with open('entropy_queries_%s.txt' % str(epoch), 'w') as f:
            for filename in queries:
                filename = filename.split('/')[-1]
                f.write(filename+'\n')

        fasterRCNN.train()
        return pool


class WeightedEntropy(nn.Module):
    def __init__(self, temp):
        super(WeightedEntropy, self).__init__()
        self.eps = 1e-10
        self.temp = temp

    def forward(self, p):
        if type(p).__module__ == np.__name__:
            p = Variable(torch.from_numpy(p).cuda())

        exponential =  torch.exp(self.temp * (1 - p))
        weight = exponential / torch.sum(exponential, dim=2).unsqueeze(2)
        
        ent = p * torch.log(p + self.eps)
        ent = torch.mul(weight, ent)

        ent = -1.0 * ent.sum(dim=-1)
        ent = ent.mean()

        ent = ent.data.cpu().numpy()
        return ent
