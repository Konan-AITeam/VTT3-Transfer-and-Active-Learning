import numpy as np
import time
from torch.autograd import Variable
import torch


class DropoutUncertainty(object):
    def __init__(self, num_query, num_dropout=20):
        super(DropoutUncertainty, self).__init__()
        self.num_query = num_query
        self.num_dropout = num_dropout

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
            im_paths = data[4][0]

            entropy = fasterRCNN(im_data, im_info, None, None, dropout_sampling=True, num_dropout=self.num_dropout)
            score = entropy

            queries_scores.append((im_paths, score))
            # print(time.time()-step_start)

        # sort
        queries_scores.sort(key=lambda element: element[1], reverse=True)
        queries_scores = queries_scores[:self.num_query]
        print(queries_scores)

        queries = [x[0] for x in queries_scores]
        scores = [x[1] for x in queries_scores]

        duration = time.time() - start
        print('query duration: %f' % duration)

        fasterRCNN.train()
        return queries, scores


class DropoutUncertainty_poolbased(object):
    def __init__(self, num_query, num_dropout=20):
        super(DropoutUncertainty_poolbased, self).__init__()
        self.num_query = num_query
        self.num_dropout = num_dropout

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
            im_paths = data[4][0]

            if im_paths in pool:
                continue

            entropy = fasterRCNN(im_data, im_info, None, None, dropout_sampling=True, num_dropout=self.num_dropout)
            score = entropy

            queries_scores.append((im_paths, score))

            if step % 100 == 0:
                print('query step: %d' % step)

        # sort
        queries_scores.sort(key=lambda element: element[1], reverse=True)
        queries_scores = queries_scores[:self.num_query]

        queries = [x[0] for x in queries_scores]
        scores = [x[1] for x in queries_scores]

        pool += queries
        print('after: %d' % len(pool))

        duration = time.time() - start
        print('query duration: %f' % duration)

        # save queries
        with open('dropout_queries_%d.txt' % epoch, 'w') as f:
            for filename in queries:
                filename = filename.split('/')[-1]
                f.write(filename+'\n')

        fasterRCNN.train()
        return pool
