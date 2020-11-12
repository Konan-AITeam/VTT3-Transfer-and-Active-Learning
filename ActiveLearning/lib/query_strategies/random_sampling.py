import numpy as np
import time


class RandomSampling(object):
    def __init__(self, num_query):
        super(RandomSampling, self).__init__()
        self.num_query = num_query

    def query(self, data_loader, fasterRCNN=None):
        queries_scores = []
        data_iter = iter(data_loader)
        iters_per_epoch = len(data_loader)
        for step in range(iters_per_epoch):
            # start = time.time()
            data = next(data_iter)
            file_name = data[4][0]
            score = np.random.randint(0, iters_per_epoch)
            queries_scores.append((file_name, score))
            # print(time.time() - start)
        # sort
        queries_scores.sort(key=lambda element: element[1])
        queries_scores = queries_scores[:self.num_query]

        queries = [x[0] for x in queries_scores]
        scores = [x[1] for x in queries_scores]
        return queries, scores


class RandomSampling_poolbased(object):
    def __init__(self, num_query):
        super(RandomSampling_poolbased, self).__init__()
        self.num_query = num_query

    def query(self, data_loader, fasterRCNN, pool, epoch=None):
        print('before: %d' % len(pool))
        data_iter = iter(data_loader)
        iters_per_epoch = len(data_loader)
        cnt = 0
        queries = []
        for step in range(iters_per_epoch):
            # start = time.time()
            data = next(data_iter)
            file_name = data[4][0]
            if file_name in pool:
                continue
            pool.append(file_name)
            queries.append(file_name)
            cnt+=1
            if cnt==self.num_query:
                break
        print('after: %d' % len(pool))

        # save queries
        with open('random_queries.txt', 'w') as f:
            for filename in queries:
                filename = filename.split('/')[-1]
                f.write(filename+'\n')

        return pool